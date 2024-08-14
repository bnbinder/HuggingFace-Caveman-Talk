venvs are optional, but highly recommended just in general
<br>
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

Make sure you have installed c++ tools from this link (for torch(!) and transformers(?))
<br>
https://visualstudio.microsoft.com/visual-cpp-build-tools/

https://developer.nvidia.com/cuda-toolkit

make sure you have pip 3.12.4 or higher cuz it complains

and install these libraries
```
pip install rich

pip install POT
pip install rouge_score
pip install gensim
pip install nltk
pip install textstat
pip install sentence-transformers

pip install sentencepiece

pip install tensorflow
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib

pip install spacy
python -m spacy download en_core_web_sm

pip install wheel
pip install ninja (idk)
pip install packaging (idk)
pip install flash-attn --no-build-isolation (idk)

pip install bitsandbytes (idk)
pip install -U "huggingface_hub[cli]" (idk)
pip install accelerate (if using low_cpu_mem_usage=true parameter when loading pretrained model ig????????!!! WHY CANT YOU JUST USE LITTLE MEMORY AT TIME >:(                  
pip install transformers==4.21.0

pip install datasets scikit-learn sentencepiece streamlit seqeval tensorboardx wandb psutil
pip install SpaCy ftfy

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (just get the latest one from their website, use cuda 121 since it complains otherwsie less verison)

pip install simpletransformers==0.61.13 (NOT USING THIS, HERE FOR AUTHOR TO REMEMBER VERSION THAT IS SUPPOSED TO WORK!!!!!!!!!!!!!!!) 
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

https://medium.com/@manuelescobar-dev implementing-and-running-llama-3-with-hugging-faces-transformers-library-40e9754d8c80

https://github.com/Dao-AILab/flash-attention

https://stackoverflow.com/questions/78746073/how-to-solve-torch-was-not-compiled-with-flash-attention-warning