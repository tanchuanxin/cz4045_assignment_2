# Word-level language modeling (FNN, LSTM, Transformer)

### Installation and Required Packages
We have already included the dataset here. All that is required is to install the python packages 
```bash 
pip install -r requirements.txt # Install required python packages
```

### Running Guide
We have simplified the model running into shell scripts. Please run these shell scripts on a GPU-enabled machine
```bash
sh batch.sh             # Trains FNN, LSTM and Transformer models with Adam Optimzer with both tied and not tied weights
sh rms_batch.sh         # Trains FNN Model with RMS Optimizer for comparison of results with Adam optimizer run in batch.sh
sh generate_text.sh     # Generate text for all models from batch.sh and rms_batch.sh  
```
