# BCI projects #
-------------------------------------------------------------------------

### Pré-Requisitos (Linux): ###

Para garantir a reprodução satisfatória dos scripts, certifique-se que seu computador possui as dependências listadas abaixo. Caso contrário, execute a sequencia de comandos para solucioná-las:

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

### Ferramentas Recomendadas: ###

* Plataforma Anaconda: https://www.anaconda.com/distribution/
* IDE Spyder (Disponível via Anaconda)
* IDE PyCharm: https://www.jetbrains.com/pycharm/promo/anaconda/

### Datasets Públicos de EEG: ###
* BCI Competition III: http://bbci.de/competition/iii/index.html

* BCI Competition IV: http://bbci.de/competition/iv/index.html

* LEE et al., 2019: http://gigadb.org/dataset/view/id/100542
