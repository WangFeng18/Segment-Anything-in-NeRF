## Viewer Intructions

### Overview
<img width="1512" alt="viewer" src="https://user-images.githubusercontent.com/44675551/235263295-6fd72175-c7ff-49e0-9760-cd608c1eeb43.png">

### NeRF Rountins

We keep all the functionalities related to NeRF itself in our viewer. Thanks again to Nerfstudio team for building such an extraordinary library.

### Point prompts
Click the "Enable SAM" checkbox, which will enable providing point prompt by clicking. The markers whill "sticks" to the object you clicked until you click "Clear SAM pins" button right under the previous one.

https://user-images.githubusercontent.com/44675551/235263730-48ca3a9a-a86e-427d-a8e8-e39ae96f00ba.mov

### Text prompts
There are two ways to use language prompts: by using a search box or filling the textbox on the config panel.

#### Textbox Prompt
You can provide text prompt by filling the text in the "Text Prompt". Text prompt will be updated once you click the "Send Text Prompt" button. One can change the threshold of relevance and the number of prompts to SAM model by filling the corresponding fields.

https://user-images.githubusercontent.com/44675551/235263775-9ca62ea3-f11e-4bc4-b179-d635c90c7d58.mov

#### Search Box Prompt
We also provide a search box for acquiring your text prompt. Press "Ctrl + F" ("^ + F" on MacOS) to toggle the search box. Once you finish inputing, press "Enter" will trigger the viewer to rerender with the given text prompt. Press "Ctrl + F" again to make the search box disappear and the original RGB image will be displayed. Futher, for convenience, the threshold and number of prompts in the right control panel is **valid** when using seach box, so you can adjust a proper parameter to fit your needs.

https://user-images.githubusercontent.com/44675551/235263849-a624874b-60b7-42d9-8d94-eb30938c501f.mov

### Switch Output
Beside the aboving, we also providing a heatmap option to visualize pixel-level relevance of the providing text prompt. You can change the "Output Render" to "clipseg_feature" to enable this.

https://user-images.githubusercontent.com/44675551/235263868-9f95a69b-af78-4f8e-a4c4-180e8f7ab38a.mov

### Fix FPS
Since the prompts are provided in the form of 2D image coordinate, the resolution will affect the final segmentation result. To solve this, we provide a checkbox to lock the resolution. With this enabled, the output resolution will not change when dragging and moving the camera, which is helpful to observing the generated segmentation mask.

## NOTE
**If you find something is not updated (e.g. enter a word in search box and press "Enter" but nothing happens), dragging the scene will probably solve the problem.** The viewer is currently work in progress, and there may exist tons of bugs. Please let us know if you encounter something unexpected, thanks in advance for you help :smiling_face_with_three_hearts:. 
