import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'
import numpy as np
import torch
import json

import time

from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar

from torch.autograd import Variable
from sklearn.utils.class_weight import compute_class_weight

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *
from datetime import datetime

# setting random seed to 1

if torch.cuda.is_available():
    #print("CUDA available, using GPU ID {}".format(config.gpu_id))
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed_all(1)
else:
    #print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

def train_one_epoch(net, optimizer, config, data_generator, 
                    epoch,  master_bar=None, tb_logger=None, wandb_logger=None, num_neighbors=20):
    
    # Set training mode
    net.train()

    # Assign parameters
    num_nodes = config.num_nodes
    #num_neighbors = np.random.choice(config.num_neighbors)#config.num_neighbors
    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch    # must be specified explicitly
    accumulation_steps = config.accumulation_steps
    # train_filepath = config.train_filepath
    # modify
    loss_type = config.loss_type
    num_neg = config.num_neg
    if loss_type == 'FL':
        gamma = config.gamma
    else:
        gamma = 0
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    # running_err_edges = 0.0
    # running_err_tour = 0.0 
    # running_err_tsp = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0

    start_epoch = time.time()
    for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
        # Generate a batch of TSPs
        try:
            batch = next(data_generator)
        except StopIteration:
            break

        # Convert batch to torch Variables
        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
        x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
        x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
        y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
        y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
        
        # Compute class weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        
        # Forward pass
        y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges,
                                    edge_cw, num_neg, loss_type, gamma)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        loss.backward()

        # Backward pass
        if (batch_num+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute error metrics and mean tour lengths
        # err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)
        pred_tour_len = mean_tour_len_edges(x_edges_values, y_preds)
        gt_tour_len = np.mean(batch.tour_len)

        # Update running data
        running_nb_data += batch_size
        running_loss += batch_size* loss.data.item()* accumulation_steps  # Re-scale loss
        # running_err_edges += batch_size* err_edges
        # running_err_tour += batch_size* err_tour
        # running_err_tsp += batch_size* err_tsp
        running_pred_tour_len += batch_size* pred_tour_len
        running_gt_tour_len += batch_size* gt_tour_len
        running_nb_batch += 1
        
        # Log intermediate statistics
        result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
            loss=running_loss/running_nb_data,
            pred_tour_len=running_pred_tour_len/running_nb_data,
            gt_tour_len=running_gt_tour_len/running_nb_data))
        master_bar.child.comment = result


        # TODO: Log to wandb (log results of this batch (not running mean))
        if wandb_logger is not None and (batch_num+1) % config.log_step == 0:
            wandb_logger.log({'avg_cost': pred_tour_len,
                              'actor_loss': loss.item(),
                              'opt_gap': pred_tour_len/gt_tour_len - 1,})


    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data
    err_edges = 0 # running_err_edges/ running_nb_data
    err_tour = 0 # running_err_tour/ running_nb_data
    err_tsp = 0 # running_err_tsp/ running_nb_data
    pred_tour_len = running_pred_tour_len/ running_nb_data
    gt_tour_len = running_gt_tour_len/ running_nb_data


    train_time = time.time()-start_epoch
    opt_gap = pred_tour_len/gt_tour_len - 1


    # Log to Tensorboard
    master_bar.write('t: ' + metrics_to_str(epoch, train_time, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len, num_neighbors))
    tb_logger.add_scalar('loss/train_loss', loss, epoch)
    tb_logger.add_scalar('pred_tour_len/train_pred_tour_len', pred_tour_len, epoch)
    tb_logger.add_scalar('optimality_gap/train_opt_gap', opt_gap, epoch)

    # return time.time()-start_epoch, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len
    return loss


def metrics_to_str(epoch, time, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len, num_neighbors=20):
    result = ( 'epoch:{epoch:0>2d}\t'
               'time:{time:.1f}h\t'
            #    'lr:{learning_rate:.2e}\t'
               'loss:{loss:.4f}\t'
               'err_edges:{err_edges:.2f}\t'
               'err_tour:{err_tour:.2f}\t'
               'err_tsp:{err_tsp:.2f}\t'
               'pred_tour_len:{pred_tour_len:.3f}\t'
               'gt_tour_len:{gt_tour_len:.3f}\t'
              'num_neighbors:{num_neighbors:0>2d}'.format(
                   epoch=epoch,
                   time=time/3600,
                #    learning_rate=learning_rate,
                   loss=loss,
                   err_edges=err_edges,
                   err_tour=err_tour,
                   err_tsp=err_tsp,
                   pred_tour_len=pred_tour_len,
                   gt_tour_len=gt_tour_len,
              num_neighbors=num_neighbors))
    return result
    
    
def test(net, config, mode, dataset, epoch,
         master_bar, tb_logger=None, wandb_logger=None, num_neighbors = 20):    
    
    # Set evaluation mode
    net.eval()

    # Assign parameters
    num_nodes = config.num_nodes
    #num_neighbors = np.random.choice(config.num_neighbors)#config.num_neighbors
    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    beam_size = config.beam_size
    # val_filepath = config.val_filepath
    # test_filepath = config.test_filepath
    # modify
    num_neg = config.num_neg
    loss_type = config.loss_type
    if loss_type == 'FL':
        gamma = config.gamma
    else:
        gamma = 0

    # Load TSP data
    batches_per_epoch = dataset.max_iter

    # Convert dataset to iterable
    dataset = iter(dataset)
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    # running_err_edges = 0.0
    # running_err_tour = 0.0
    # running_err_tsp = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0
    
    with torch.no_grad():
        start_test = time.time()
        for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
            # Generate a batch of TSPs
            try:
                batch = next(dataset)
            except StopIteration:
                break

            # Convert batch to torch Variables
            x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
            x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
            x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
            x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
            y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
            y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
            
            # Compute class weights (if uncomputed)
            if type(edge_cw) != torch.Tensor:
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

            # Forward pass --- modify
            y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, 
                                        edge_cw, num_neg, loss_type, gamma)
            loss = loss.mean()  # Take mean of loss across multiple GPUs

            # Compute error metrics
            # err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)

            # Get batch beamsearch tour prediction
            if mode == 'val':  # Validation: faster 'vanilla' beamsearch
                bs_nodes = beamsearch_tour_nodes(
                    y_preds, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')
            elif mode == 'test':  # Testing: beamsearch with shortest tour heuristic 
                bs_nodes = beamsearch_tour_nodes_shortest(
                    y_preds, x_edges_values, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')
            
            # Compute mean tour length
            pred_tour_len = mean_tour_len_nodes(x_edges_values, bs_nodes)
            gt_tour_len = np.mean(batch.tour_len)

            # Update running data
            running_nb_data += batch_size
            running_loss += batch_size* loss.data.item()
            # running_err_edges += batch_size* err_edges
            # running_err_tour += batch_size* err_tour
            # running_err_tsp += batch_size* err_tsp
            running_pred_tour_len += batch_size* pred_tour_len
            running_gt_tour_len += batch_size* gt_tour_len
            running_nb_batch += 1

            # Log intermediate statistics
            result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
                loss=running_loss/running_nb_data,
                pred_tour_len=running_pred_tour_len/running_nb_data,
                gt_tour_len=running_gt_tour_len/running_nb_data))
            master_bar.child.comment = result
            
            

    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data
    err_edges = 0 # running_err_edges/ running_nb_data
    err_tour = 0 # running_err_tour/ running_nb_data
    err_tsp = 0 # running_err_tsp/ running_nb_data
    pred_tour_len = running_pred_tour_len/ running_nb_data
    gt_tour_len = running_gt_tour_len/ running_nb_data
    opt_gap = pred_tour_len/gt_tour_len - 1

    val_time = time.time()- start_test

    master_bar.write('v: ' + metrics_to_str(epoch, val_time, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len, num_neighbors))
    tb_logger.add_scalar('loss/val_loss', loss, epoch)
    tb_logger.add_scalar('pred_tour_len/val_pred_tour_len', pred_tour_len, epoch)
    tb_logger.add_scalar('optimality_gap/val_opt_gap', opt_gap, epoch)

    if wandb_logger is not None:
        wandb_logger.log({'val_avg_reward': pred_tour_len, 'epoch': epoch})


    return loss, pred_tour_len

    
def main(config, data_generator, wandb_logger=None):
    # Instantiate the network
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)  
    
    pretrained = config.pretrained  # False
    patience = config.patience  # 1
    lr_scale = config.lr_scale  # 1.
    pretrained_path = config.pretrained_path    # None
    var_neighbor = config.var_neighbor      # 5
    random_neighbor = config.random_neighbor    # True
    netname = config.netname    # 'att-gcn'
    
    if netname == 'att-gcn':
        net = ResidualGatedGCNModel(config, dtypeFloat, dtypeLong)
    else:
        raise ValueError('Network name not recognized')
    
    net = nn.DataParallel(net)

    if torch.cuda.is_available():
        net.cuda()
    if pretrained:
        if pretrained_path is not None:
            log_dir = pretrained_path
            if torch.cuda.is_available():
                checkpoint = torch.load(log_dir)
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            log_dir = f"./tsp-models/{config.expt_name}/"
            if torch.cuda.is_available():
                checkpoint = torch.load(log_dir+"best_val_checkpoint.tar")
            net.load_state_dict(checkpoint['model_state_dict'])
    print(net)

    # Compute number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)
 
    # Create log directory
    log_dir = config.log_dir
    save_dir = config.save_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    json.dump(vars(config), open(f"{save_dir}/config.json", "w"), indent=4)
    tb_logger = SummaryWriter(log_dir)  # Define Tensorboard writer

    # Training parameters
    #batch_size = config.batch_size
    #batches_per_epoch = config.batches_per_epoch
    #accumulation_steps = config.accumulation_steps
    #num_nodes = config.num_nodes
    #num_neighbors = config.num_neighbors
    max_epochs = config.max_epochs
    val_every = config.val_every
    test_every = config.test_every
    learning_rate = config.learning_rate * lr_scale
    decay_rate = config.decay_rate
    num_patience = 0
    val_loss_old = 1e6  # For decaying LR based on validation loss
    val_loss_best = 1e6
    best_pred_tour_len = 1e6  # For saving checkpoints

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#     optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
#                                 momentum=0.9, weight_decay=0.0005)
    print(optimizer)
    

    # load validate dataset
    val_dataset = GoogleTSPReader(
        num_nodes=config.num_nodes, num_neighbors=config.num_neighbors,
        batch_size=config.batch_size, filepath=config.val_filepaths, shuffled=False
    )

    epoch_bar = master_bar(range(max_epochs))
    for epoch in epoch_bar:
        # Log to Tensorboard
        if random_neighbor:
            if epoch%var_neighbor==0:
                num_neighbors = np.random.choice(config.num_neighbors)
        else:
            num_neighbors = config.num_neighbors
        tb_logger.add_scalar('learning_rate', learning_rate, epoch)
        
        # Train
        train_loss = train_one_epoch(net, optimizer, config, data_generator, epoch, 
                        epoch_bar, tb_logger, wandb_logger, num_neighbors)


        if epoch % val_every == 0 or epoch == max_epochs-1:
            # Validate
            val_loss, val_pred_tour_len = test(net, config, 'val', val_dataset, epoch,
                            epoch_bar, tb_logger, wandb_logger, num_neighbors = num_neighbors)
            
            # Save checkpoint
            if val_pred_tour_len < best_pred_tour_len:
                best_pred_tour_len = val_pred_tour_len  # Update best prediction
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir,"best_val_checkpoint_{}.tar".format(epoch)))
            
            # Update learning rate
            if val_loss > 0.99 * val_loss_old:
                learning_rate /= decay_rate
                optimizer = update_learning_rate(optimizer, learning_rate)
            
            val_loss_old = val_loss  # Update old validation loss
            # Early Stopping
            if val_loss_best > val_loss:
                num_patience = 0
                val_loss_best = val_loss
            else:
                num_patience +=1
        

#         if epoch % test_every == 0 or epoch == max_epochs-1:
#             # Test
#             test_time, test_loss, test_err_edges, test_err_tour, test_err_tsp, test_pred_tour_len, test_gt_tour_len = test(net, config, epoch_bar, mode='test')
#             epoch_bar.write('T: ' + metrics_to_str(epoch, test_time, learning_rate, test_loss, test_err_edges, test_err_tour, test_err_tsp, test_pred_tour_len, test_gt_tour_len))
#             writer.add_scalar('loss/test_loss', test_loss, epoch)
#             writer.add_scalar('pred_tour_len/test_pred_tour_len', test_pred_tour_len, epoch)
#             writer.add_scalar('optimality_gap/test_opt_gap', test_pred_tour_len/test_gt_tour_len - 1, epoch)
        
        # Save training checkpoint at the end of epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(save_dir,"last_train_checkpoint.tar"))
        
        # Save checkpoint after every 250 epochs
        if epoch != 0 and (epoch % config.checkpoint_epochs == 0 or epoch == max_epochs-1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir,f"checkpoint_epoch{epoch}.tar"))
        if num_patience >= patience:
            pass # TODO: forbid early stop
            break
        
    return net

