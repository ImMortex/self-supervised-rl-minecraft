# Screenshots for Observation Test
One screenshots of an agent is needed to test the observation, which depends on the screen resolution.
Please set the resolution ``1920x1080`` for the main screen of your device.
Please insert one screenshots of your main screen that was made using the ``make_screenshot_main_screen.py``.

To make a screenshot you need an opened Minecraft instance as explained in the chapter Setup Minecraft settings of ``README.md`` 
in the project root.

The <b>screenshot</b> should contain the following:
- the Minecraft inventory should be <b>closed</b> and should contain 45 pieces of the target item as shown in the ``example.png``
- the Minecraft health bar and hunger bar should be full 
- the Minecraft experience bar and experience level should be 0
Save the made screenshot in this folder as ``screenshot.png``

After this run the observation test: 
````shell
python run_agent_observation_test.py
````

If the test failed, create a new config named ``inventoryScreenshotNoFullscreenConfCustom.json`` as a copy of ``inventoryScreenshotNoFullscreenConf.json``
inside the ``config`` folder. Adapt the values and repeat the test until a successful mapping is created. 
- Hint: Use an image editor to count the pixels of screenshot.png relative to the upper left corner of the img to find the correct mapping.
