#Eamon's resnet for valuing a gamestate of Quoridor
import logging
import argparse
import sys
import torch
import torch.nn as nn

from StateNode import QuoridorDataset

class QuoridorValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input structure:
        # Channel 1: 9x9 - P1 pawn location (binary)
        # Channel 2: 9x9 - P2 pawn location (binary)
        # Channel 3: 8x8 - Horizontal walls (binary) 
        # Channel 4: 8x8 - Vertical walls (binary)

        #pawn positions
        self.pawn_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Process wall positions (8x8)
        self.wall_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Residual tower for pawn features
        self.pawn_residual = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32)
            ) for _ in range(3)
        ])

        # Residual tower for wall features
        self.wall_residual = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32)
            ) for _ in range(3)
        ])

        # Feature combination layer
        self.combine_features = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),  # 1x1 conv to merge channels
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),  # 8x8 = 64 features after flattening
            nn.ReLU()
        )

        # Meta features processing (walls remaining + turn indicator)
        self.meta_features = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU()
        )

        # Final evaluation combining all features
        self.final_evaluation = nn.Sequential(
            nn.Linear(256 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, pawn_state, wall_state, meta_state):
        # pawn_state: batch x 2 x 9 x 9
        # wall_state: batch x 2 x 8 x 8 (horizontal and vertical walls)
        # meta_state: batch x 3 (walls remaining for each player + turn indicator)

        # Process pawn positions
        x_pawns = self.pawn_conv(pawn_state)
        for res_block in self.pawn_residual:
            identity = x_pawns
            x_pawns = res_block(x_pawns)
            x_pawns += identity
            x_pawns = torch.relu(x_pawns)

        # Process wall positions
        x_walls = self.wall_conv(wall_state)
        for res_block in self.wall_residual:
            identity = x_walls
            x_walls = res_block(x_walls)
            x_walls += identity
            x_walls = torch.relu(x_walls)

        # Downsample pawn features to 8x8 to match wall features
        x_pawns = nn.functional.interpolate(x_pawns, size=(8, 8), mode='bilinear')

        # Combine pawn and wall features
        x_combined = torch.cat([x_pawns, x_walls], dim=1)
        x_combined = self.combine_features(x_combined)

        # Process through value head
        value_features = self.value_head(x_combined)

        # Process meta features
        meta_features = self.meta_features(meta_state)

        # Combine all features for final evaluation
        combined = torch.cat([value_features, meta_features], dim=1)
        final_value = self.final_evaluation(combined)

        return final_value

def train_step(model, optimizer, data_batch):
    model.train()
    pawn_states, wall_states, meta_states, target_values = data_batch

    if torch.cuda.is_available():
            pawn_states = pawn_states.cuda()
            wall_states = wall_states.cuda()
            meta_states = meta_states.cuda()
            target_values = target_values.cuda()

    optimizer.zero_grad()
    predicted_values = model(pawn_states, wall_states, meta_states)
    loss = nn.MSELoss()(predicted_values, target_values)

    loss.backward()
    optimizer.step()

    return loss.item()

def train(model, tree_file : str, batch_size : int, num_epochs: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    if torch.cuda.is_available():
        model.cuda()

    #while tree can reliably fit in memory, just get all batches up front
    batches = []
    with QuoridorDataset(tree_file, batch_size) as dataset:
        for batch in dataset.generate_batches():
            batches.append(batch)
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch in batches:
            loss = train_step(model, optimizer, batch)
            epoch_loss += loss
            num_batches += 1

            if num_batches % 100 == 0:
                logging.info(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss:.4f}")

        if num_batches == 0:
            logging.error("Num batches is 0, not training")
            return
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)

        logging.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train Quoridor Value Network')
    parser.add_argument('--load-tree', type=str, dest="load_tree", help='Path to load search tree')
    parser.add_argument('--load-model', type=str, dest="load_model", help='Path to load model weights')
    parser.add_argument('--save-model', type=str, dest="save_model", help='Path to save model weights')
    parser.add_argument('--export', dest="export", help='Will save inferencable version of model and exit', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, dest="batch_size", default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--log-level', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    model = QuoridorValueNet()
    if args.load_model:
        logging.info(f"Loading {args.load_model}")
        model.load_state_dict(torch.load(args.load_model))

        if args.export:
            model.eval()

            # Create example inputs for tracing
            example_pawn_state = torch.zeros(1, 2, 9, 9)
            example_wall_state = torch.zeros(1, 2, 8, 8)
            example_meta_state = torch.zeros(1, 3)

            # Use TorchScript to create a serializable version
            traced_script_module = torch.jit.trace(model, (example_pawn_state, example_wall_state, example_meta_state))

            # Save the model
            traced_script_module.save(args.save_model)
            sys.exit(0)


    train(model, args.load_tree, args.batch_size, args.epochs)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)

if __name__ == "__main__":
    main()

