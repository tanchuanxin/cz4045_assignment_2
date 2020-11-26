python generate.py --cuda --checkpoint ./models/model-FNN-tied.pt
python generate.py --cuda --checkpoint ./models/model-FNN-not_tied.pt

python generate.py --cuda --checkpoint ./models/model-rms-FNN-tied.pt
python generate.py --cuda --checkpoint ./models/model-rms-FNN-not_tied.pt

python generate.py --cuda --checkpoint ./models/model-Transformer-tied.pt
python generate.py --cuda --checkpoint ./models/model-Transformer-not_tied.pt

python generate.py --cuda --checkpoint ./models/model-LSTM-tied.pt
python generate.py --cuda --checkpoint ./models/model-LSTM-not_tied.pt



