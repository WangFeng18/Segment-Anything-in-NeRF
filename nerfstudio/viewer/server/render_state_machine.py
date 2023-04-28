# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This file contains the render state machine, which is responsible for deciding when to render the image """
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from copy import deepcopy

import torch
from typing_extensions import Literal, get_args

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.utils import writer
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfstudio.viewer.viser.messages import CameraMessage

if TYPE_CHECKING:
    from nerfstudio.viewer.server.viewer_state import ViewerState

import numpy as np
from time import time
from collections import deque

RenderStates = Literal["low_move", "low_static", "high"]
RenderActions = Literal["rerender", "move", "static", "step"]

def get_prompt_points(cam_msg, image_height: int, image_width: int):
    xs = np.array(cam_msg.xs)
    ys = np.array(cam_msg.ys)
    xs = (xs * image_width).astype(np.int32)
    ys = (ys * image_height).astype(np.int32)

    # ret shape: [n_pts, 2] : int
    return np.stack([xs, ys], axis=-1)

@dataclass
class RenderAction:
    """Message to the render state machine"""

    action: RenderActions
    """The action to take """
    cam_msg: CameraMessage
    """The camera message to render"""
    use_fixed_fps: bool = False
    use_text_prompt: bool = False
    text_prompt: str = ""
    threshold: float = 1.0


class RenderStateMachine(threading.Thread):
    """The render state machine is responsible for deciding how to render the image.
    It decides the resolution and whether to interrupt the current render.

    Args:
        viewer: the viewer state
    """

    def __init__(self, viewer: ViewerState):
        threading.Thread.__init__(self)
        self.transitions: Dict[RenderStates, Dict[RenderActions, RenderStates]] = {
            s: {} for s in get_args(RenderStates)
        }
        # by default, everything is a self-transition
        for a in get_args(RenderActions):
            for s in get_args(RenderStates):
                self.transitions[s][a] = s
        # then define the actions between states
        self.transitions["low_move"]["static"] = "low_static"
        self.transitions["low_static"]["static"] = "high"
        self.transitions["low_static"]["step"] = "high"
        self.transitions["low_static"]["move"] = "low_move"
        self.transitions["high"]["move"] = "low_move"
        self.transitions["high"]["rerender"] = "low_static"
        self.next_action: Optional[RenderAction] = None
        self.state: RenderStates = "low_static"
        self.render_trigger = threading.Event()
        self.target_fps = 24
        self.viewer = viewer
        self.interrupt_render_flag = False
        self.last_cam_msg = None
        self.daemon = True
        self.render_times = deque([], maxlen=3)

    def action(self, action: RenderAction):
        """Takes an action and updates the state machine

        Args:
            action: the action to take
        """
        """Takes an action and updates the state machine

        Args:
            action: the action to take
        """
        if self.next_action is None:
            self.next_action = action
        elif action.action == "step" and (
            self.state == "low_move" or self.next_action.action in ("move", "static", "rerender")
        ):
            # ignore steps if:
            #  1. we are in low_moving state
            #  2. the current next_action is move, static, or rerender
            # For debug
            return
        elif self.next_action == "rerender":
            # never overwrite rerenders
            pass
        else:
            #  monimal use case, just set the next action
            self.next_action = action

        # handle interrupt logic
        if self.state in ["high", "low_static"] and self.next_action.action in ("move", "rerender"):
            self.interrupt_render_flag = True
        self.render_trigger.set()

    def _render_img(self, action: RenderAction):
        """Takes the current camera, generates rays, and renders the iamge

        Args:
            cam_msg: the camera message to render
        """

        print("into render")
        print(action.cam_msg)
        cam_msg = action.cam_msg if action.cam_msg is not None else self.last_cam_msg

        viewer_utils.update_render_aabb(
            crop_viewport=self.viewer.control_panel.crop_viewport,
            crop_min=self.viewer.control_panel.crop_min,
            crop_max=self.viewer.control_panel.crop_max,
            model=self.viewer.get_model(),
        )

        image_height, image_width = self._calculate_image_res(cam_msg.aspect if cam_msg is not None else self.last_cam_msg.aspect)

        intrinsics_matrix, camera_to_world_h = get_intrinsics_matrix_and_camera_to_world_h(
            cam_msg, image_height=image_height, image_width=image_width
        )

        camera_to_world = camera_to_world_h[:3, :]
        camera_to_world = torch.stack(
            [
                camera_to_world[0, :],
                camera_to_world[2, :],
                camera_to_world[1, :],
            ],
            dim=0,
        )

        camera_type_msg = cam_msg.camera_type
        if camera_type_msg == "perspective":
            camera_type = CameraType.PERSPECTIVE
        elif camera_type_msg == "fisheye":
            camera_type = CameraType.FISHEYE
        elif camera_type_msg == "equirectangular":
            camera_type = CameraType.EQUIRECTANGULAR
        else:
            camera_type = CameraType.PERSPECTIVE

        camera = Cameras(
            fx=intrinsics_matrix[0, 0],
            fy=intrinsics_matrix[1, 1],
            cx=intrinsics_matrix[0, 2],
            cy=intrinsics_matrix[1, 2],
            camera_type=camera_type,
            camera_to_worlds=camera_to_world[None, ...],
            times=torch.tensor([self.viewer.control_panel.time], dtype=torch.float32),
        )
        camera = camera.to(self.viewer.get_model().device)

        # breakpoint()

        with self.viewer.train_lock if self.viewer.train_lock is not None else contextlib.nullcontext():
            camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=self.viewer.get_model().render_aabb)

            with TimeWriter(None, None, write=False) as vis_t:
                self.viewer.get_model().eval()
                step = self.viewer.step
                if self.viewer.control_panel.crop_viewport:
                    color = self.viewer.control_panel.background_color
                    if color is None:
                        background_color = torch.tensor([0.0, 0.0, 0.0], device=self.viewer.pipeline.model.device)
                    else:
                        background_color = torch.tensor(
                            [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0],
                            device=self.viewer.get_model().device,
                        )
                    with background_color_override_context(background_color), torch.no_grad():
                        outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                        self.viewer.get_model().train()
                else:
                    with torch.no_grad():
                        # breakpoint()
                        points = None
                        text_prompt = None
                        threshold = 0.0
                        topk = 0
                        if self.viewer.use_sam:
                            points = get_prompt_points(cam_msg, image_height, image_width)
                            self.viewer.n_points_sam = len(points)
                            print("SAM case\n")
                            # outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle, points=points, intrin=intrinsics_matrix, c2w=camera_to_world)
                        if self.viewer.use_text_prompt:
                            print("Text Prompt SAM case\n")
                            print("text prompts:", self.viewer.text_prompt)
                            print("threshold:", self.viewer.threshold)
                            print("topK:", self.viewer.topk)
                            text_prompt = self.viewer.text_prompt
                            threshold = self.viewer.threshold
                            topk = int(self.viewer.topk)

                            # outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle, text_prompt=self.viewer.text_prompt, threshold=self.viewer.threshold)
                            # outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle) 

                        if self.viewer.use_search_text:
                            print("Seach Text Case\n")
                            text_prompt = self.viewer.search_text
                            points = None
                            threshold = self.viewer.control_panel.threshold
                            topk = int(self.viewer.control_panel.topk)
                            
                        outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle, points=points, intrin=intrinsics_matrix, c2w=camera_to_world, text_prompt=text_prompt, topk=topk, thresh=threshold) 
                        print(outputs.keys())
                        
                        self.viewer.get_model().train()
                self.viewer.get_model().train()
        num_rays = len(camera_ray_bundle)
        render_time = vis_t.duration
        writer.put_time(
            name=EventName.VIS_RAYS_PER_SEC, duration=num_rays / render_time, step=step, avg_over_steps=True
        )
        self.viewer.viser_server.send_status_message(eval_res=f"{image_height}x{image_width}px", step=step)
        return outputs

    def run(self):
        """Main loop for the render thread"""
        while True:
            self.render_trigger.wait()
            self.render_trigger.clear()
            action = self.next_action
            # assert action is not None, "Action should never be None at this point"
            # TODO check this workaround
            if action is None:
                continue
            self.next_action = None
            if self.state == "high" and action.action == "static":
                # if we are in high res and we get a static action, we don't need to do anything
                # TODO currently a workaround
                if self.last_cam_msg is not None and action.cam_msg is not None and len(action.cam_msg.xs) != len(self.last_cam_msg.xs):
                    pass
                else:
                    continue
                # pass
            self.state = self.transitions[self.state][action.action]
            # breakpoint()
            try:
                # breakpoint()
                with viewer_utils.SetTrace(self.check_interrupt):
                    t = time()
                    outputs = self._render_img(action)
                    t = time() - t
                    print("+"*30)
                    print(f"FPS: {1./t:.3f}")
                    print("+"*30)
                    self.render_times.append(t)
                    self.viewer.fps = len(self.render_times) / sum(self.render_times)
                    self.viewer.viser_server.update_fps(self.viewer.fps)
                    if action.cam_msg is not None:
                        self.last_cam_msg = action.cam_msg
            except viewer_utils.IOChangeException:
                # breakpoint()
                # if we got interrupted, don't send the output to the viewer
                print("Error and Error and Error")
                continue
            except Exception as e:
                # breakpoint()
                print("Someother error")
            # breakpoint()
            self._send_output_to_viewer(outputs)
            # if we rendered a static low res, we need to self-trigger a static high-res
            if self.state == "low_static":
                self.action(RenderAction("static", action.cam_msg))
            # breakpoint()

    def check_interrupt(self, frame, event, arg):  # pylint: disable=unused-argument
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.interrupt_render_flag:
                self.interrupt_render_flag = False
                raise viewer_utils.IOChangeException
        return self.check_interrupt

    def _send_output_to_viewer(self, outputs: Dict[str, Any]):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the model
        """
        self.viewer.control_panel.update_output_options(list(outputs.keys()))

        output_render = self.viewer.control_panel.output_render
        self.viewer.update_colormap_options(
            dimensions=outputs[output_render].shape[-1], dtype=outputs[output_render].dtype
        )
        selected_output = (viewer_utils.apply_colormap(self.viewer.control_panel, outputs) * 255).type(torch.uint8)

        self.viewer.viser_server.set_background_image(
            selected_output.cpu().numpy(),
            file_format=self.viewer.config.image_format,
            quality=self.viewer.config.jpeg_quality,
        )

    def _calculate_image_res(self, aspect_ratio: float) -> Tuple[int, int]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            apect_ratio: the aspect ratio of the current view
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        max_res = self.viewer.control_panel.max_res
        if self.state == "high":
            # high res is always static
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        elif self.state in ("low_move", "low_static"):
            if EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                vis_rays_per_sec = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
            else:
                vis_rays_per_sec = 100000
            target_fps = self.target_fps
            num_vis_rays = vis_rays_per_sec / target_fps
            image_height = (num_vis_rays / aspect_ratio) ** 0.5
            image_height = int(round(image_height, -1))
            image_height = max(min(max_res, image_height), 30)
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        else:
            raise ValueError(f"Invalid state: {self.state}")

        if self.viewer.use_fixed_fps:
            image_height = self.viewer.control_panel.max_res
            image_width = int(image_height * aspect_ratio)

        return image_height, image_width
