/path/to/python.exe -m venv venv_name
C:\Users\Camper\AppData\Local\Programs\Python\Python38\python.exe -m venv venv_name

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned

venv_name/Scripts/activate

https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install transformers==4.21.0
pip install torch torchvision torchaudio
pip install datasets scikit-learn sentencepiece streamlit seqeval tensorboardx wandb psutil
pip install simpletransformers==0.61.13 
(NOT USING THIS, HERE FOR AUTHOR TO REMEMBER VERSION THAT IS SUPPOSED TO WORK)
pip intall SpaCy ftfy

venv_name/Scripts/deactivate

Set-ExecutionPolicy -ExecutionPolicy Undefined






https://code.visualstudio.com/docs/python/environments

https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4

https://github.com/huggingface/optimum/issues/344

https://stackoverflow.com/questions/8949252/why-do-i-get-attributeerror-nonetype-object-has-no-attribute-something