# Rizzanator 3000

Rizzanator 3000 is an AI-powered assistant that provides real-time conversational advice and flirty responses to help users keep conversations engaging.

## Features

- Real-time live listening and advice.
- Message assistance for crafting flirty responses.
- On-device processing for privacy and offline usage.

## Installation

### Requirements

- Python 3.7 or higher
- PyTorch
- Transformers
- TensorFlow
- TensorFlow Lite (for mobile conversion)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rizzanator_project.git
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the `training` directory to train the model.

## Usage

- **Training the Model**:

  ```bash
  python training/train_model.py
  ```

- **Converting the Model for Mobile**:

  ```bash
  python model_conversion/convert_model.py
  ```

- **Integrating into Mobile App**:

  - Follow the instructions in the `mobile_app/android` and `mobile_app/ios` directories.

## License

MIT License
