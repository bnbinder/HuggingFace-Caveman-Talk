# What Is This
to get a better understanding and introduction to NLP, decided to do a pet project about it. You need a gpu to run this, since the glove and llama model need one to function.

# What To Install
venvs are optional, but highly recommended just in general
<br>
<br>
To create your virtual enviornment, open the terminal in your repository folder and enter
```
/path/to/python.exe -m venv venv_name
```
<br>
For the word movers distance, install one of the gloves and extract the zip into the repository. then edit the `gloveInputFile` String in the code with the path to the glove model of your choosing
<br>
<br>
https://nlp.stanford.edu/projects/glove/
<br>
<br>
Make sure you have installed c++ tools from this link
<br>
<br>
https://visualstudio.microsoft.com/visual-cpp-build-tools/
<br>
<br>
And install the nvidia cuda toolkit
<br>
<br>
https://developer.nvidia.com/cuda-toolkit
<br>
<br>

run the `requirements.txt` file

and run this command to install the spacy english language model
```
python -m spacy download en_core_web_sm
```

# Running The Code
Open an admin access terminal and run this code so you can activate the virtual enviornment
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

Then open the terminal in your repository folder and enter
```
venv_name/Scripts/activate
```
<br>
Run the python file like usual, and follow the directions
<br>
<br>

Once your done running your code, run this in a terminal inside your repository folder
```
venv_name/Scripts/deactivate
```

And run this in an admin access terminal
```
Set-ExecutionPolicy -ExecutionPolicy Undefined
```

# Sources and Help
https://medium.com/@manuelescobar-dev/implementing-and-running-llama-3-with-hugging-faces-transformers-library-40e9754d8c80
