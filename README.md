To create your virtual enviornment, open the terminal in your repository folder and enter
```
/path/to/python.exe -m venv venv_name
```

Then open an admin access terminal and run this code so you can activate the virtual enviornment
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

Then open the terminal in your repository folder and enter
```
venv_name/Scripts/activate
```

Make sure you have installed c++ tools from this link
<br>
https://visualstudio.microsoft.com/visual-cpp-build-tools/

```
pip install transformers==4.21.0
pip install torch torchvision torchaudio
pip install datasets scikit-learn sentencepiece streamlit seqeval tensorboardx wandb psutil
pip install SpaCy ftfy
(NOT USING THIS, HERE FOR AUTHOR TO REMEMBER VERSION THAT IS SUPPOSED TO WORK!!!!!!!!!!!!!!!) pip install simpletransformers==0.61.13
```

Once your done running your code, run this in a terminal inside your repository folder
```
venv_name/Scripts/deactivate
```

And run this in an admin access terminal
```
Set-ExecutionPolicy -ExecutionPolicy Undefined
```

# Sources and Help
https://code.visualstudio.com/docs/python/environments

https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4

https://github.com/huggingface/optimum/issues/344

https://stackoverflow.com/questions/8949252/why-do-i-get-attributeerror-nonetype-object-has-no-attribute-something
