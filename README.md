### BCI projects ###
-------------------------------------------------------------------------

## Atenção: 

* A fim de garantir a execução dos scripts sem erros. Certifique-se que seu computador já possui as dependências listadas abaixo. Caso contrário, execute a sequencia de comandos a seguir para solucioná-las:

```shell

sudo apt-get install sox
sudo apt-get update
sudo apt-get install python3.7
sudo apt-get install python-pip
pip install numpy scipy matplotlib hyperopt
pip install -U sklearn
pip install https://github.com/kivy/kivy
pip install kivy-garden
garden install bar
garden install graph
pip install -U https://api.github.com/repos/mne-tools/mne-python/zipball/master
sudo apt-get install xclip xsel
sudo chmod 777 /dev/input/event* 

```

## Public datasets:
BCI Competition III: http://bbci.de/competition/iii/index.html

BCI Competition IV: http://bbci.de/competition/iv/index.html
