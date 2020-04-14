# BCI projects #
-------------------------------------------------------------------------
* ./overmind/ &nbsp; :: Overmind: a MI-based BCI Platform
* ./dsformat/ &nbsp;&nbsp; :: Scripts para normalização de dados públicos de EEG (Formato para Overmind)
* ./scripts/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :: Scripts python de laboratório (cenários independentes)

### Demo: 
- https://www.youtube.com/watch?v=5l7inGVSwsc&list=PLnBWXhrTR44Tfd_fE-lSsUCVP8Rp-DOMU

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

* Sistema Operacional Linux (derivados Debian)
* Plataforma Anaconda: https://www.anaconda.com/distribution/
* IDE Spyder (Disponível via Anaconda)
* IDE PyCharm: https://www.jetbrains.com/pycharm/promo/anaconda/

### Dados Públicos (EEG): ###
* BCI Competition III: http://bbci.de/competition/iii/index.html
* BCI Competition IV: http://bbci.de/competition/iv/index.html
* LEE et al., 2019: http://gigadb.org/dataset/view/id/100542

* Dados III3a, III4a, IV2a e IV2b devidamente compatibilizados com o Overmind disponíveis <a href="https://iftoedubr-my.sharepoint.com/:u:/g/personal/vitorvilasboas_ifto_edu_br/EUNu9fhzsUBJudJuNybEX38B2-xhEln8z0SZjUau0XI3ag?e=FtWXp1" target="blank">aqui</a>.
