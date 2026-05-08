@echo on
echo Building Conda environment 'genai_project'...
call conda env create -f conda\build\requirements.yaml
call conda activate genai_project
call pip3 uninstall -y torch torchvision torchaudio pytorch
call pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
call pip3 uninstall pillow -y
call pip3 install pillow
call pip3 install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu130
echo Build complete.
pause