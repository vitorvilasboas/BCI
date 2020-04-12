# BCI projects #
-------------------------------------------------------------------------
* .\overmind\ &emsp;&emsp;&emsp; >> Overmind: a MI-based BCI Platform
* .\scrips\ &emsp;&emsp;&emsp;&emsp;&nbsp; >> Scripts de laboratório (independentes da plataforma Overmind)
* .\linux_dataset_format\ &emsp; >> Scripts para compatibilizar o formato dos arquivos dos conjuntos de dados públicos de EEG

### Demo: 
- https://youtu.be/5l7inGVSwsc

### Pré-Requisitos (Linux): ###

Para garantir a reprodução satisfatória dos scripts, certifique-se que seu computador possui as dependências listadas abaixo. Caso contrário, execute a sequência de comandos para solucioná-las:

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

### Dados Públicos (EEG): ###
* BCI Competition III: http://bbci.de/competition/iii/index.html
* BCI Competition IV: http://bbci.de/competition/iv/index.html
* LEE et al., 2019: http://gigadb.org/dataset/view/id/100542
