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

""" Manage the state of the viewer """
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from rich import box, style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.server.control_panel import ControlPanel
from nerfstudio.viewer.server.gui_utils import get_viewer_elements
from nerfstudio.viewer.server.render_state_machine import (
    RenderAction,
    RenderStateMachine,
)
from nerfstudio.viewer.server.viewer_elements import ViewerElement
from nerfstudio.viewer.viser import ViserServer
from nerfstudio.viewer.viser.messages import (
    CameraMessage,
    CameraPathOptionsRequest,
    CameraPathPayloadMessage,
    ClearSamPinsMessage,
    CropParamsMessage,
    NerfstudioMessage,
    SamMessage,
    SaveCheckpointMessage,
    TimeConditionMessage,
    TrainingStateMessage,
    TextPromptMessage,
    ThresholdMessage,
    FPSMessage,
    SearchTextMessage,
)

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer

CONSOLE = Console(width=120)


@decorate_all([check_main_thread])
class ViewerState:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
        pipeline: pipeline object to use
        trainer: trainer object to use

    Attributes:
        viewer_url: url to open viewer
    """

    viewer_url: str

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        trainer: Optional[Trainer] = None,
        train_lock: Optional[threading.Lock] = None,
    ):
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        self.viewer_url = viewer_utils.get_viewer_url(websocket_port)
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("HTTP", f"[link={self.viewer_url}][blue]{self.viewer_url}[/link]")

        CONSOLE.print(Panel(table, title="[bold][yellow]Viewer[/bold]", expand=False))

        self.include_time = self.pipeline.datamanager.includes_time

        self.sam_capable = self.pipeline.model.sam_capable
        self.text_prompt_capable = self.pipeline.model.text_prompt_capable
        
        self.use_sam = False
        self.use_text_prompt = False
        self.use_fixed_fps = False
        self.use_search_text = False
        
        self.text_prompt = ""
        self.threshold = 0.0
        self.topk = 5

        self.search_text = None

        self.fps = -1.0

        # viewer specific variables
        self.output_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = "training"
        self._prev_train_state: Literal["training", "paused", "completed"] = "training"

        self.camera_message = None

        self.viser_server = ViserServer(host="127.0.0.1", port=websocket_port)

        self.viser_server.register_handler(TrainingStateMessage, self._handle_training_state_message)
        self.viser_server.register_handler(SaveCheckpointMessage, self._handle_save_checkpoint)
        self.viser_server.register_handler(CameraMessage, self._handle_camera_update)
        self.viser_server.register_handler(CameraPathOptionsRequest, self._handle_camera_path_option_request)
        self.viser_server.register_handler(CameraPathPayloadMessage, self._handle_camera_path_payload)
        self.viser_server.register_handler(CropParamsMessage, self._handle_crop_params_message)
        if self.include_time:
            self.viser_server.use_time_conditioning()
            self.viser_server.register_handler(TimeConditionMessage, self._handle_time_condition_message)

        self.viser_server.register_handler(SamMessage, self._handle_sam_message)
        self.viser_server.register_handler(ClearSamPinsMessage, self._handle_clear_sam_pins_message)
        self.viser_server.register_handler(SearchTextMessage, self._handle_search_text_message)

        self.control_panel = ControlPanel(
            self.include_time,
            self._interrupt_render,
            self._crop_params_update,
            self._output_type_change,
            self.sam_capable,
            self._sam_update,
            self._clear_sam_pins,
            self.text_prompt_capable,
            self._send_text_prompt,
            self._clear_text_prompt,
            self._fixed_fps_cb,
        )
        self.control_panel.install(self.viser_server)

        def nested_folder_install(folder_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [self._interrupt_render(element), prev_cb(element)]
            else:
                with self.viser_server.gui_folder(folder_labels[0]):
                    nested_folder_install(folder_labels[1:], element)

        self.viewer_elements = get_viewer_elements(self.pipeline)
        for param_path, element in self.viewer_elements:
            folder_labels = param_path.split("/")[:-1]
            nested_folder_install(folder_labels, element)

        self.render_statemachine = RenderStateMachine(self)
        self.render_statemachine.start()

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _interrupt_render(self, _) -> None:
        """Interrupt current render."""
        if self.camera_message is not None:
            self.render_statemachine.action(RenderAction("rerender", self.camera_message))

    def _crop_params_update(self, _) -> None:
        """Update crop parameters"""
        self.render_statemachine.action(RenderAction("rerender", self.camera_message))
        crop_min = torch.tensor(self.control_panel.crop_min, dtype=torch.float32)
        crop_max = torch.tensor(self.control_panel.crop_max, dtype=torch.float32)
        scene_box = SceneBox(aabb=torch.stack([crop_min, crop_max], dim=0))
        self.viser_server.update_scene_box(scene_box)
        crop_scale = crop_max - crop_min
        crop_center = crop_max + crop_min
        self.viser_server.send_crop_params(
            crop_enabled=self.control_panel.crop_viewport,
            crop_bg_color=self.control_panel.background_color,
            crop_scale=tuple(crop_scale.tolist()),
            crop_center=tuple(crop_center.tolist()),
        )

    def _sam_update(self, _) -> None:
        self.viser_server.use_sam(self.control_panel.use_sam)
        if self.control_panel.use_sam:
            self.control_panel.output_render = "masked_rgb"
            self.use_sam = True
        else:
            self.viser_server.clear_sam_pins()
            if not self.use_text_prompt:
                self.control_panel.output_render = "rgb"
            self.use_sam = False
        self.use_sam = self.control_panel.use_sam
        self.render_statemachine.action(action=RenderAction("static", None))

    def _clear_sam_pins(self, _) -> None:
        self.viser_server.clear_sam_pins()
        # self.control_panel.output_render = "rgb"
        self.render_statemachine.action(action=RenderAction("static", None))

    def _fixed_fps_cb(self, _) -> None:
        self.use_fixed_fps = self.control_panel.use_fixed_fps

    # def _update_text_prompt(self, _) -> None:
    #     self.viser_server.update_text_prompt(self.control_panel.text_prompt)

    # def _update_threshold(self, _) -> None:
    #     self.viser_server.update_threshold(self.control_panel.threshold)
    #     self.threshol

    def _send_text_prompt(self, _) -> None:
        # TODO check here
        # Fuck this is beautiful
        self.use_text_prompt = True
        self.control_panel.output_render = "text_prompt"
        self.control_panel.output_render = "masked_rgb"
        self.text_prompt = self.control_panel.text_prompt
        self.threshold = self.control_panel.threshold
        self.topk = self.control_panel.topk
        self.render_statemachine.action(action=RenderAction("static", None))
        
    def _clear_text_prompt(self, _) -> None:
        self.text_prompt = ""
        if not self.use_sam:
            self.control_panel.output_render = "rgb"
        self.control_panel.text_prompt = ""
        self.use_text_prompt = False
        self.render_statemachine.action(action=RenderAction("static", None))

    def _handle_training_state_message(self, message: NerfstudioMessage) -> None:
        """Handle training state message from viewer."""
        assert isinstance(message, TrainingStateMessage)
        self.train_btn_state = message.training_state
        self.training_state = message.training_state
        self.viser_server.set_training_state(message.training_state)

    def _handle_save_checkpoint(self, message: NerfstudioMessage) -> None:
        """Handle save checkpoint message from viewer."""
        assert isinstance(message, SaveCheckpointMessage)
        if self.trainer is not None:
            self.trainer.save_checkpoint(self.step)

    def _handle_camera_update(self, message: NerfstudioMessage) -> None:
        """Handle camera update message from viewer."""
        assert isinstance(message, CameraMessage)
        self.camera_message = message
        if message.is_moving:
            self.render_statemachine.action(RenderAction("move", self.camera_message))
            if self.training_state == "training":
                self.training_state = "paused"
        else:
            self.render_statemachine.action(RenderAction("static", self.camera_message))
            self.training_state = self.train_btn_state

    def _handle_camera_path_option_request(self, message: NerfstudioMessage) -> None:
        """Handle camera path option request message from viewer."""
        assert isinstance(message, CameraPathOptionsRequest)
        camera_path_dir = self.datapath / "camera_paths"
        if camera_path_dir.exists():
            all_path_dict = {}
            for path in camera_path_dir.iterdir():
                if path.suffix == ".json":
                    all_path_dict[path.stem] = load_from_json(path)
            self.viser_server.send_camera_paths(all_path_dict)

    def _handle_camera_path_payload(self, message: NerfstudioMessage) -> None:
        """Handle camera path payload message from viewer."""
        assert isinstance(message, CameraPathPayloadMessage)
        camera_path_filename = message.camera_path_filename + ".json"
        camera_path = message.camera_path
        camera_paths_directory = self.datapath / "camera_paths"
        camera_paths_directory.mkdir(parents=True, exist_ok=True)
        write_to_json(camera_paths_directory / camera_path_filename, camera_path)

    def _handle_crop_params_message(self, message: NerfstudioMessage) -> None:
        """Handle crop parameters message from viewer."""
        assert isinstance(message, CropParamsMessage)
        self.control_panel.crop_viewport = message.crop_enabled
        self.control_panel.background_color = message.crop_bg_color
        center = np.array(message.crop_center)
        scale = np.array(message.crop_scale)
        crop_min = center - scale / 2.0
        crop_max = center + scale / 2.0
        self.control_panel.crop_min = tuple(crop_min.tolist())
        self.control_panel.crop_max = tuple(crop_max.tolist())

    def _handle_time_condition_message(self, message: NerfstudioMessage) -> None:
        """Handle time conditioning message from viewer."""
        assert isinstance(message, TimeConditionMessage)
        self.control_panel.time = message.time

    def _handle_sam_message(self, message: SamMessage) -> None:
        self.control_panel.use_sam = message.use_sam

    def _handle_clear_sam_pins_message(self, message: ClearSamPinsMessage) -> None:
        # TODO finish
        # self.render_statemachine.action()
        pass

    def _handle_text_prompt_message(self, message: TextPromptMessage):
        self.control_panel.text_prompt = message.text_prompt

    def _handle_threshold_message(self, message: ThresholdMessage):
        self.control_panel.threshold = message.threshold

    def _handle_search_text_message(self, message: SearchTextMessage):
        print("+" * 20)
        print(message.text)
        print(message.switch_to_heat_map)
        print("+" * 20)
        if message.switch_to_heat_map:
            print("use heat map")
            self.render_output_before = self.control_panel.output_render if self.control_panel.output_render != "clipseg_feature" else "rgb"
            self.use_search_text = True
            # TODO: add here the real render output for search heatmap
            self.control_panel.output_render = "clipseg_feature"
            self.search_text = message.text
        else:
            # for disable search text and back to the previous mode
            self.use_search_text = False
            assert getattr(self, "render_output_before", None) is not None, "original mode should be stored, this is bug and should be report"
            print(f"previous render: {self.render_output_before}")
            self.control_panel.output_render = self.render_output_before
            self.search_text = None

    @property
    def training_state(self) -> Literal["training", "paused", "completed"]:
        """Get training state flag."""
        if self.trainer is not None:
            return self.trainer.training_state
        return self.train_btn_state

    @training_state.setter
    def training_state(self, training_state: Literal["training", "paused", "completed"]) -> None:
        """Set training state flag."""
        if self.trainer is not None:
            self.trainer.training_state = training_state

    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    def init_scene(self, dataset: InputDataset, train_state=Literal["training", "paused", "completed"]) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        self.viser_server.send_file_path_info(
            config_base_dir=self.log_filename.parents[0],
            data_base_dir=self.datapath,
            export_path_name=self.log_filename.parent.stem,
        )

        # draw the training cameras and images
        image_indices = self._pick_drawn_image_idxs(len(dataset))
        for idx in image_indices:
            image = dataset[idx]["image"]
            bgr = image[..., [2, 1, 0]]
            camera_json = dataset.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
            self.viser_server.add_dataset_image(idx=f"{idx:06d}", json=camera_json)

        # draw the scene box (i.e., the bounding box)
        self.viser_server.update_scene_box(dataset.scene_box)

        # set the initial state whether to train or not
        self.train_btn_state = train_state
        self.viser_server.set_training_state(train_state)

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        # # breakpoint()
        self.step = step

        if self.camera_message is None:
            return

        if (
            self.trainer is not None
            and self.trainer.training_state == "training"
            and self.control_panel.train_util != 1
        ):
            if (
                EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
                train_util = self.control_panel.train_util
                vis_n = self.control_panel.max_res**2
                train_n = num_rays_per_batch
                train_time = train_n / train_s
                vis_time = vis_n / vis_s

                render_freq = train_util * vis_time / (train_time - train_util * train_time)
            else:
                render_freq = 30
            if step > self.last_step + render_freq:
                self.last_step = step
                # print(self.camera_message.xs)
                # print(self.camera_message.ys)
                # print("here")
                # breakpoint()
                # TODO modify here
                self.render_statemachine.action(RenderAction("step", self.camera_message))

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_type_changed:
            self.control_panel.update_colormap_options(dimensions, dtype)
            self.output_type_changed = False

    def get_model(self) -> Model:
        """Returns the model."""
        return self.pipeline.model

    def training_complete(self) -> None:
        """Called when training is complete."""
        self.training_state = "completed"
        self.viser_server.set_training_state("completed")
