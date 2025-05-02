import os
import torch
import copy
from ultralytics import YOLO

# Function to aggregate model parameters
def parameter_aggregation(model_ensemble):
    state_dicts = [m.model.state_dict() for m in model_ensemble]
    combined_state = {}
    
    for param_key in state_dicts[0]:
        param_stack = torch.stack([state_dicts[i][param_key] for i in range(len(model_ensemble))])
        param_stack = param_stack.to(torch.float16)
        combined_state[param_key] = torch.mean(param_stack, dim=0)
        
    return combined_state

# Training configuration
CLIENT_EPOCHS = 2          # Local training iterations per client
TOTAL_COMM_ROUNDS = 150    # Total federated learning rounds
PARTICIPANT_COUNT = 3      # Number of participating clients

BASE_DIR = os.getcwd()
CONFIG_FILE = os.path.join(BASE_DIR, 'config.yaml')

# Model storage setup
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'model_weights')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Client configuration files
CLIENT_CONFIGS = (
    os.path.join(BASE_DIR, 'a.yaml'),
    os.path.join(BASE_DIR, 'b.yaml'),
    os.path.join(BASE_DIR, 'c.yaml')
)

if __name__ == '__main__':
    # Initialize base model
    global_model = YOLO('yolo11n.yaml').load('yolo11n.pt')

    # Federated learning process
    for comm_cycle in range(TOTAL_COMM_ROUNDS):
        print(f"\n=== Federation Round {comm_cycle} ===\n")
        
        # Create client models
        client_models = [copy.deepcopy(global_model) for _ in range(PARTICIPANT_COUNT)]

        # Client-side training
        for client_id, local_model in enumerate(client_models):
            local_model.train(
                data=CLIENT_CONFIGS[client_id],
                epochs=CLIENT_EPOCHS,
                imgsz=640,
                batch=16,
                save=True,
                resume=True,
                iou=0.5,
                conf=0.001,
                plots=False,
                workers=0,
                augment=False  # Disable data augmentation
            )

        # Model aggregation
        aggregated_params = parameter_aggregation(client_models)
        global_model.model.load_state_dict(aggregated_params)

        # Save intermediate model
        model_checkpoint = f'fl_weights_e{CLIENT_EPOCHS}_c{PARTICIPANT_COUNT}_r{comm_cycle}.pt'
        save_path = os.path.join(MODEL_SAVE_DIR, model_checkpoint)
        torch.save(global_model, save_path)

    # Save final aggregated model
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'final_fl_model.pt')
    torch.save(global_model, final_model_path)