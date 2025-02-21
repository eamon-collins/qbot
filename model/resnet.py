#Eamon's resnet for valuing a gamestate of Quoridor
import torch
import torch.nn as nn

class QuoridorValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 9x9x3 tensor 
        # Channel 1: P1 pawn location (binary)
        # Channel 2: P2 pawn location (binary)  
        # Channel 3: Wall locations (binary)

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.residual_tower = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32)
            ) for _ in range(4)  # 4 residual blocks should be sufficient for Quoridor
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(81, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Additional meta features
        self.meta_features = nn.Sequential(
            nn.Linear(2, 8),  # Input: [p1_walls, p2_walls]
            nn.ReLU()
        )

        # Combine board evaluation with meta features
        self.final_evaluation = nn.Sequential(
            nn.Linear(256 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, board_state, meta_state):
        # board_state: batch x 3 x 9 x 9
        # meta_state: batch x 2 (walls remaining for each player)
        
        x = self.conv_block(board_state)
        
        # Residual connections
        for res_block in self.residual_tower:
            identity = x
            x = res_block(x)
            x += identity
            x = torch.relu(x)

        # Process value head
        value_features = self.value_head[:-1](x)  # Everything except final tanh
        
        # Process meta features
        meta_features = self.meta_features(meta_state)
        
        # Combine features
        combined = torch.cat([value_features, meta_features], dim=1)
        final_value = self.final_evaluation(combined)

        return final_value

def train_step(model, optimizer, data_batch):
    model.train()
    board_states, meta_states, target_values = data_batch
    
    if torch.cuda.is_available():
            board_states = board_states.cuda()
            meta_states = meta_states.cuda()
            target_values = target_values.cuda()

    optimizer.zero_grad()
    predicted_values = model(board_states, meta_states)
    loss = nn.MSELoss()(predicted_values, target_values)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train(model, dataset: QuoridorDataset, num_epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    model.cuda()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataset.generate_batches():
            loss = train_step(model, optimizer, batch)
            epoch_loss += loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                logging.info(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        logging.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train Quoridor Value Network')
    parser.add_argument('--load-tree', type=str, dest="load_tree", help='Path to load search tree')
    parser.add_argument('--load-model', type=str, dest="load_model", help='Path to load model weights')
    parser.add_argument('--save-model', type=str, dest="save_model", help='Path to save model weights')
    parser.add_argument('--batch-size', type=int, dest="batch_size", default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--log-level', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    model = QuoridorValueNet()
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    
    with QuoridorDataset(args.load_tree, args.batch_size) as dataset:
        train(model, dataset, args.epochs)
    
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)

if __name__ == "__main__":
    main()

