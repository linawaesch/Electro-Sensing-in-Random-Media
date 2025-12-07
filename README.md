# Electro-Sensing-in-Random-Media
The code base implements the full pipeline: it simulates multi frequency snapshots in random media, learns and reorders the sensing matrix, and then uses it for GPT reconstruction and shape classification experiments.


How to run the pipeline example:

python snapshot_generation.py  
python recover_unordered_G.py  
python order.py  
python Figure_5_shape_identification_vs_noise.py
