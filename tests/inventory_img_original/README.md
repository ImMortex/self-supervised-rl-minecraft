Minecraft can be run in fullscreen or no fullscreen. In both cases, the agent takes screenshots to evaluate them. 
With no fullscreen, a defined marging is cut off and the image is scaled to 1920x1080 (like fullscreen). 
In order to be able to cut out and evaluate individual image areas, a mapping was defined for both cases. 

To test the mapping, corresponding images are entered in ``inventory_img_marked_mapping`` dir  and output marked in color in ``inventory_img_original`` dir.

Run ``create_inventory_slots_mapping.py`` in the project root to run the test.
A human must manually evaluate the results and change the mapping files in ``config`` dir.