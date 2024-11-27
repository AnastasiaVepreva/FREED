# FREED++

This repository is the official Pytorch implementation of "FREED++: Improving RL Agents for Fragment-Based Molecule Generation by Thorough Reproduction".

### Зависимости
Package  | Version
--- | ---
Python | 3.7.12
PyTorch | 1.12.1
TorchVision | 0.13.1
CUDA | 11.3.1
DGL | 0.9.1.post1
RDKit | 2020.09.1.0

### Установка окружения
```bash
conda env create -f environment.yml
conda activate freedpp
```

## Обучение
### FREED++
```bash
python main.py     --exp_root ../experiments     --alert_collections ../alert_collections.csv     --fragments ../zinc_crem.json     --receptor ../protein.pdbqt     --vina_program ./env/qvina02     --starting_smile "c1([*:1])c([*:2])ccc([*:3])c1"     --fragmentation crem     --num_sub_proc 12     --n_conf 1     --exhaustiveness 1     --save_freq 50     --epochs 200     --commands "train,sample"     --reward_version soft     --box_center "x y z"     --box_size "x,y,z"     --seed 150     --name freedpp     --objectives "Ic50,SA,DockingScore"     --weights "1.0,1.0,1.0" --num_mols 10000

```
## Оценка свойств
```bash
python main.py     --exp_root ../experiments     --alert_collections ../alert_collections.csv     --fragments ../zinc_crem.json     --receptor ../protein.pdbqt     --vina_program ./env/qvina02     --starting_smile "c1([*:1])c([*:2])ccc([*:3])c1"     --fragmentation crem     --num_sub_proc 12     --n_conf 1     --exhaustiveness 1     --save_freq 50     --epochs 200     --commands "evaluate"     --reward_version soft     --box_center "x y z"     --box_size "x,y,z"     --seed 150     --name freedpp     --objectives "Ic50,SA,DockingScore"     --weights "1.0,1.0,1.0" --checkpoint ..experiments/freedpp/ckpt/model_200.pth
```
## Пояснения к командам:
--receptor здесь нужно указать имя файла с белком
--box_center координаты центра связывания
--box_size размер ограничивающего бокса
--objectives названия свойств SA и DockingScore остаются без изменений для любого кейса. Ic50: Альцгеймер - Ic50_alzh, склероз - Ic50_sklr, рак легких - Ic50_lung, лекарственная устойчивость - Ic50_resist, дислипидемия - Ic50_dslp, Паркинсон - ic50_park
--starting_smile можно поставить любой стартовый фрагмент молекулы. По умолчанию стоит фрагмент бензольного кольца
