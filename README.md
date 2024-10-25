# ConvTSMixer, MlpTSMixer and TsT

Time Series Extension of "Patches Are All You Need"

Implementations of some prototypical time series mixers based on Conv, MLP, and ViT archs. modified for the probabilistic multivariate forecasting use case, where the emission head is currently an "independent same-family" distribution, e.g., diagonal Student-T.

In everything that follows, the inputs are typically 4-Tensors of shape `[Batch, Variate-dim, Context-length, 1+Features]`, and during training, the subsequent prediction window values are given `[B, Variate-dim, Pred-length]`. The inputs are embedded via 2d-conv to obtain patch embeddings:

![Screenshot 2024-10-25 at 09 29 21](https://github.com/user-attachments/assets/5ecc92c5-a115-44a1-95de-971d2d34ed58)

## ConvTSMixer

![Screenshot 2024-10-25 at 09 29 55](https://github.com/user-attachments/assets/2033b75b-6105-4c5d-bfc6-aaf9c8330d5d)


## MlpTSMixer

![Screenshot 2024-10-25 at 09 30 09](https://github.com/user-attachments/assets/14db4edd-2004-4152-acc7-3c926c97c306)

## TsT (ViT style)

![Screenshot 2024-10-25 at 09 30 40](https://github.com/user-attachments/assets/6a5f88d4-8ded-41fe-9cc5-3ce1ab9df5b9)

## Output head

![Screenshot 2024-10-25 at 09 31 12](https://github.com/user-attachments/assets/324bec4c-d4bf-40ea-9084-413d1c647870)
