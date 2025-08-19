## Computer Vision-Powered Hand Gesture-Based Keyboard Controller

This is a basic implementation of using hand gestures to control the computer keyboard (10 FPS).

`output/efficientnet_b0.pth` has the trained parameters of the used model. Specifically, it is the EfficientNet B0 model architecture, trained on a chosen set of 12 hand gestures of the [HaGRIDv2](https://github.com/hukenovs/hagrid/tree/Hagrid_v2-1M) dataset.

`key_mappings.json` specifies which hand gesture corresponds to which key on the keyboard. 

The current implementation only supports single key presses, which is triggered by the detection of some hand gesture. To press the same key again, one must change into the specific gesture again, e.g. peace sign to press 'w', lower my hand to reset gesture, make a peace sign again to press 'w' for a second time.

**How to run:**

1. Configure `key_mappings.json`
2. Run `python main.py`
3. Press 'q' while in the pop-up window to quit
