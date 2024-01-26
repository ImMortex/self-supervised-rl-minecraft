# Setup und Cleanup Minecraft Agent on PC
- Description how to set up the agent on a PC
- The Description uses ``D:`` as example dir.

## MultiMC Minecraft
- java 20 zip in D: unpack and select in MultiMC GUI
- login to Minecraft account
- Install and start Minecraft 1.19.1
- create new World: Use standard values, except difficulty: ``Peaceful`` and generate single biom: ``Forest``


## setup project (before python env setup)
- Set OS resolution to 1920x1080 so that uniform screenshots can be taken (Minecraft Inventory position)
- Clone projekt https://gitlab.hs-anhalt.de/ki/projekte/minecraft-rl/self-supervised-rl-minecraft in D: 
- do ``Initial project setup`` explained in README.md if not done

## setup python env
- Anaconda prompt öffnen

Navigate to project folder
````shell
D:
cd D:\path\to\your\project\self-supervised-rl-minecraft
````

(One-time) Import conda python environment. Ensures that torch with cuda is executable on gpu.

````shell
conda env create --prefix D:/conda/ma_christian_gurski_pc_pool -f ma_christian_gurski.yaml
conda activate D:\conda\ma_christian_gurski_pc_pool
````
If necessary, delete the old env beforehand:
````shell
conda deactivate D:\conda\ma_christian_gurski_pc_pool
conda remove -n D:\conda\ma_christian_gurski_pc_pool --all
rmdir D:\conda\ma_christian_gurski_pc_pool
````

Test should deliver ``True``
````shell
python cuda_check.py
````

(Always) Install other requirements
````shell
pip install -r ./requirements.txt
````

## setup project (after python env setup)
- select conda python intepreter previously created in IDE: D:\conda\ma_christian_gurski_pc_pool\python.exe
- Run agent for training and to record its transitions for pretraining:
````shell
python agent_train_a3c_async.py
````



## Clean up (after final use of the PC)
- Create env backup if necessary: Export: conda env export > ma_christian_gurski_pc_pool.yaml
- Close Minecraft and IDE
- Reset OS resolution to recommended setting
- Back up data
- In D: Delete folder MA_Christian_Gurski_Agent_on_Pool_PC and self-supervised-rl-minecraft
