# Word-level language modeling (FNN, LSTM, Transformer)

```bash 
pip install -r requirements.txt # Install required python packages
sh batch.sh             # Trains FNN, LSTM and Transformer models with Adam Optimzer with both tied and not tied weights
sh rms_batch.sh         # Trains FNN Model with RMS Optimizer for comparison of results with Adam optimizer run in batch.sh
sh generate_text.sh     # Generate text for all models from batch.sh and rms_batch.sh  
```

