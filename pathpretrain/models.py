import time
import os
import sys
import seaborn as sns
from .schedulers import Scheduler
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.mobilenet import MobileNetV2
from sklearn.metrics import classification_report, f1_score
#from apex import amp
import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from kornia.losses import DiceLoss
import  kornia.losses as kl
import tqdm


matplotlib.use('Agg')
sns.set()


class MLP(nn.Module):
    """Multi-layer perceptron model.

    Parameters
    ----------
    n_input:int
            Number input dimensions.
    hidden_topology:list
            List of hidden topology
    dropout_p:float
            Amount dropout.
    n_outputs:int
            Number outputs.
    binary:bool
            Binary output with sigmoid transform.
    softmax:bool
            Whether to apply softmax on output.

    """

    def __init__(self, n_input, hidden_topology, dropout_p, n_outputs=1, binary=True, softmax=False):
        super(MLP, self).__init__()
        self.topology = [n_input] + hidden_topology + [n_outputs]
        layers = [nn.Linear(self.topology[i], self.topology[i + 1])
                  for i in range(len(self.topology) - 2)]
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
        self.layers = [nn.Sequential(layer, nn.LeakyReLU(
        ), nn.Dropout(p=dropout_p)) for layer in layers]
        self.output_layer = nn.Linear(self.topology[-2], self.topology[-1])
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        if binary:
            output_transform = nn.Sigmoid()
        elif softmax:
            output_transform = nn.Softmax()
        else:
            output_transform = nn.Dropout(p=0.)
        self.layers.append(nn.Sequential(self.output_layer, output_transform))
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.mlp(x)

class AuxNet(nn.Module):
    def __init__(self,net,n_aux_features):
        super().__init__()
        self.net=net
        self.features=self.net.features
        self.output=self.net.output
        self.n_features=self.net.output.in_features
        self.n_aux_features=n_aux_features
        self.transform_nn=nn.Sequential(nn.Linear(self.n_aux_features,self.n_features),nn.LeakyReLU())
        self.gate_nn=MLP(self.n_features,[32],dropout_p=0.2,binary=False)#nn.Linear(self.n_features,1)

    def forward(self,x,z=None):
        x=self.features(x)
        x = x.view(x.size(0), -1)
        if z is not None:
            z=self.transform_nn(z)
            #print(x.shape,z.shape,self.gate_nn(x).shape,self.gate_nn(z).shape)
            gate_h=F.softmax(torch.cat([self.gate_nn(xz) for xz in [x,z]],1),1)
            x = gate_h[:,0].unsqueeze(1) * x + gate_h[:,1].unsqueeze(1) * z
        x = self.output(x)
        return x

def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_cuda=False,
                  use_data_parallel=True,
                  net_extra_kwargs=None,
                  load_ignore_extra=False,
                  num_classes=None,
                  in_channels=3,
                  remap_to_cpu=True,
                  remove_module=False,
                  semantic_segmentation=False,
                  n_aux_features=None,
                  encoderweights="imagenet"):
    from pytorchcv.model_provider import get_model
    import segmentation_models_pytorch as smp
    """ https://raw.githubusercontent.com/osmr/imgclsmob/master/pytorch/utils.py
        Create and initialize model by name.

        Parameters
        ----------
        model_name : str
        Model name.
        use_pretrained : bool
        Whether to use pretrained weights.
        pretrained_model_file_path : str
        Path to file with pretrained weights.
        use_cuda : bool
        Whether to use CUDA.
        use_data_parallel : bool, default True
        Whether to use parallelization.
        net_extra_kwargs : dict, default None
        Extra parameters for model.
        load_ignore_extra : bool, default False
        Whether to ignore extra layers in pretrained model.
        num_classes : int, default None
        Number of classes.
        in_channels : int, default None
        Number of input channels.
        remap_to_cpu : bool, default False
        Whether to remape model to CPU during loading.
        remove_module : bool, default False
        Whether to remove module from loaded model.

        Returns
        -------
        Module
        Model.
        """
    kwargs = {"pretrained": use_pretrained}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    if not semantic_segmentation:
        if kwargs['pretrained']:
            kwargs['pretrained']=False
            net = get_model(model_name, **kwargs)
            net_shape_dict = {k:v.shape for k,v in net.state_dict().items()}
            kwargs['num_classes']=1000
            kwargs['pretrained']=True
            net_pretrained=get_model(model_name, **kwargs).state_dict()
            net.load_state_dict({k:v for k,v in net_pretrained.items() if v.shape==net_shape_dict[k]},strict=False)
        else:
            net = get_model(model_name, **kwargs)

        if n_aux_features is not None:
            net=AuxNet(net,n_aux_features)

    else:
        net = smp.Unet(model_name,encoder_weights=encoderweights, classes=num_classes, in_channels=in_channels)

    return net


def generate_model(architecture, num_classes, semantic_segmentation, pretrained=False, n_aux_features=None,encoderweights="imagenet"):
    #    from pytorchcv.pytorch.utils import prepare_model
    if os.path.exists(architecture):
        model = torch.load(architecture,map_location='cpu')
    else:
        model = prepare_model(architecture,
                          use_pretrained=pretrained,
                          pretrained_model_file_path='',
                          use_cuda=False,
                          num_classes=num_classes,
                          semantic_segmentation=semantic_segmentation,
                          n_aux_features=n_aux_features,
                          encoderweights=encoderweights)
    return model


class ModelTrainer:
    """Trainer for the neural network model that wraps it into a scikit-learn like interface.

    Parameters
    ----------
    model:nn.Module
            Deep learning pytorch model.
    n_epoch:int
            Number training epochs.
    validation_dataloader:DataLoader
            Dataloader of validation dataset.
    optimizer_opts:dict
            Options for optimizer.
    scheduler_opts:dict
            Options for learning rate scheduler.
    loss_fn:str
            String to call a particular loss function for model.
    reduction:str
            Mean or sum reduction of loss.
    num_train_batches:int
            Number of training batches for epoch.
    """

    def __init__(self, model, n_epoch=300, validation_dataloader=None, optimizer_opts=dict(name='adam', lr=1e-3, weight_decay=1e-4), scheduler_opts=dict(scheduler='warm_restarts', lr_scheduler_decay=0.5, T_max=10, eta_min=5e-8, T_mult=2), loss_fn='ce', reduction='mean', num_train_batches=None, opt_level='O1', checkpoints_dir='checkpoints',tensor_dataset=False,transforms=None,semantic_segmentation=False,save_metric='loss',save_after_n_batch=0):

        self.model = model
        # self.amp_handle = amp.init(enabled=True)
        optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
        loss_functions = {'bce': nn.BCEWithLogitsLoss(reduction=reduction), 'ce': nn.CrossEntropyLoss(
            reduction=reduction), 'mse': nn.MSELoss(reduction=reduction), 
            'nll': nn.NLLLoss(reduction=reduction),'dice':DiceLoss(),'custom':CustomLoss(), 
            'kbfll':kl.BinaryFocalLossWithLogits(alpha=0.25),'kfocal':kl.FocalLoss(alpha=0.5),
            'ktvl':kl.TverskyLoss(alpha=0.5, beta=0.5),
            'dice1':DiceLoss1(),'dicebce':DiceBCELoss()}
        if 'name' not in list(optimizer_opts.keys()):
            optimizer_opts['name'] = 'adam'
        self.optimizer = optimizers[optimizer_opts.pop('name')](
            self.model.parameters(), **optimizer_opts)
        if False and torch.cuda.is_available():
            self.cuda = True
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=opt_level)
        else:
            self.cuda = False
        self.scheduler = Scheduler(
            optimizer=self.optimizer, opts=scheduler_opts)
        self.n_epoch = n_epoch
        self.validation_dataloader = validation_dataloader
        self.loss_fn = loss_functions[loss_fn]
        self.loss_fn_name = loss_fn
        self.bce = (self.loss_fn_name == 'bce')
        self.sigmoid = nn.Sigmoid()
        self.original_loss_fn = copy.deepcopy(loss_functions[loss_fn])
        self.num_train_batches = num_train_batches
        self.val_loss_fn = copy.deepcopy(loss_functions[loss_fn])
        self.verbosity=0
        self.checkpoints_dir=checkpoints_dir
        self.tensor_dataset=tensor_dataset
        self.transforms=transforms
        self.semantic_segmentation=semantic_segmentation
        self.save_metric=save_metric
        self.save_after_n_batch=save_after_n_batch
        self.train_batch_count=0
        self.initial_seed=0
        self.seed=0

    def save_checkpoint(self,model,epoch,batch=0):
        os.makedirs(self.checkpoints_dir,exist_ok=True)
        out_name = f"{batch}.batch" if batch else f"{epoch}.epoch"
        torch.save(model,os.path.join(self.checkpoints_dir,f"{out_name}.checkpoint.pth"))

    def calc_loss(self, y_pred, y_true):
        """Calculates loss supplied in init statement and modified by reweighting.

        Parameters
        ----------
        y_pred:tensor
                Predictions.
        y_true:tensor
                True values.

        Returns
        -------
        loss

        """

        return self.loss_fn(y_pred, y_true)

    def calc_val_loss(self, y_pred, y_true):
        """Calculates loss supplied in init statement on validation set.

        Parameters
        ----------
        y_pred:tensor
                Predictions.
        y_true:tensor
                True values.

        Returns
        -------
        val_loss

        """

        return self.val_loss_fn(y_pred, y_true)

    def reset_loss_fn(self):
        """Resets loss to original specified loss."""
        self.loss_fn = self.original_loss_fn

    def add_class_balance_loss(self, y, custom_weights=''):
        """Updates loss function to handle class imbalance by weighting inverse to class appearance.

        Parameters
        ----------
        dataset:DynamicImageDataset
                Dataset to balance by.

        """
        self.class_weights = compute_class_weight('balanced',np.unique(y),y)#dataset.get_class_weights() if not custom_weights else np.array(
            #list(map(float, custom_weights.split(','))))
        if custom_weights:
            self.class_weights = self.class_weights / sum(self.class_weights)
        print('Weights:', self.class_weights)
        self.original_loss_fn = copy.deepcopy(self.loss_fn)
        weight = torch.tensor(self.class_weights, dtype=torch.float)
        if torch.cuda.is_available():
            weight = weight.cuda()
        if self.loss_fn_name == 'ce':
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        elif self.loss_fn_name == 'nll':
            self.loss_fn = nn.NLLLoss(weight=weight)
        else:  # modify below for multi-target
            self.loss_fn = lambda y_pred, y_true: sum([self.class_weights[i] * self.original_loss_fn(
                y_pred[y_true == i], y_true[y_true == i]) if sum(y_true == i) else 0. for i in range(2)])

    def calc_best_confusion(self, y_pred, y_true):
        """Calculate confusion matrix on validation set for classification/segmentation tasks, optimize threshold where positive.

        Parameters
        ----------
        y_pred:array
                Predictions.
        y_true:array
                Ground truth.

        Returns
        -------
        float
                Optimized threshold to use on test set.
        dataframe
                Confusion matrix.

        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        threshold = thresholds[np.argmin(
            np.sum((np.array([0, 1]) - np.vstack((fpr, tpr)).T)**2, axis=1)**.5)]
        y_pred = (y_pred > threshold).astype(int)
        return threshold, pd.DataFrame(confusion_matrix(y_true, y_pred), index=['F', 'T'], columns=['-', '+']).iloc[::-1, ::-1].T

    def loss_backward(self, loss):
        """Backprop using mixed precision for added speed boost.

        Parameters
        ----------
        loss:loss
                Torch loss calculated.

        """
        # with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
        # 	scaled_loss.backward()
        # loss.backward()
        if self.cuda:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    # @pysnooper.snoop('train_loop.log')
    def train_loop(self, epoch, train_dataloader):
        """One training epoch, calculate predictions, loss, backpropagate.

        Parameters
        ----------
        epoch:int
                Current epoch.
        train_dataloader:DataLoader
                Training data.

        Returns
        -------
        float
                Training loss for epoch

        """
        self.model.train(True)
        running_loss = 0.
        n_batch = len(
            train_dataloader.dataset) // train_dataloader.batch_size if self.num_train_batches == None else self.num_train_batches
        for i, batch in enumerate(train_dataloader):
            starttime = time.time()
            X, y_true = batch[:2]
            if len(batch)==3: Z=batch[2]
            else: Z=None

            if i == n_batch:
                break

            # X = Variable(batch[0], requires_grad=True)
            # y_true = Variable(batch[1])

            if torch.cuda.is_available():
                X = X.cuda()
                y_true = y_true.cuda()
                if Z is not None: Z=Z.cuda()

            if self.tensor_dataset:
                if self.semantic_segmentation: X,y_true=self.transforms['train'](X,y_true)
                else: X=self.transforms['train'](X)

            y_pred = self.model(X) if Z is None else self.model(X,Z)
            # y_true=y_true.argmax(dim=1)

            loss = self.calc_loss(y_pred, y_true)  # .view(-1,1)
            train_loss = loss.item()
            running_loss += train_loss
            self.optimizer.zero_grad()
            self.loss_backward(loss)  # loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            endtime = time.time()
            if self.verbosity >=1:
                print("Epoch {}[{}/{}] Time:{}, Train Loss:{}".format(epoch,
                                                                  i, n_batch, round(endtime - starttime, 3), train_loss))
            self.train_batch_count+=1
            if self.save_after_n_batch and self.train_batch_count%self.save_after_n_batch==0:
                val_loss,val_f1=self.val_loop(epoch, self.val_dataloader)
                self.batch_val_losses.append(val_loss)
                self.batch_val_f1.append(val_f1)
                self.save_best_val_model(val_loss, val_f1, self.batch_val_losses, self.batch_val_f1, epoch, True, self.train_batch_count)
                self.model.train(True)

        self.scheduler.step()
        running_loss /= n_batch
        return running_loss

    def val_loop(self, epoch, val_dataloader, print_val_confusion=True, save_predictions=True):
        """Calculate loss over validation set.

        Parameters
        ----------
        epoch:int
                Current epoch.
        val_dataloader:DataLoader
                Validation iterator.
        print_val_confusion:bool
                Calculate confusion matrix and plot.
        save_predictions:int
                Print validation results.

        Returns
        -------
        float
                Validation loss for epoch.
        """
        self.model.train(False)
        n_batch = len(val_dataloader.dataset) // val_dataloader.batch_size
        running_loss = 0.
        Y = {'pred': [], 'true': []}
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                # X = Variable(batch[0], requires_grad=True)
                # y_true = Variable(batch[1])
                X, y_true = batch[:2]
                if len(batch)==3: Z=batch[2]
                else: Z=None
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_true = y_true.cuda()
                    if Z is not None: Z=Z.cuda()

                if self.tensor_dataset:
                    if self.semantic_segmentation: X,y_true=self.transforms['val'](X,y_true)
                    else: X=self.transforms['val'](X)

                y_pred = self.model(X) if Z is None else self.model(X,Z)
                # y_true=y_true.argmax(dim=1)
                # if save_predictions:
                Y['true'].append(
                    y_true.detach().cpu().numpy().astype(int).flatten())
                y_pred_numpy = ((y_pred if self.bce else self.sigmoid(
                    y_pred)).detach().cpu().numpy()).astype(float)
                if self.loss_fn_name in ['ce','dice','dice1','dicebce','custom']:
                    y_pred_numpy = y_pred_numpy.argmax(axis=1)
                Y['pred'].append(y_pred_numpy.flatten())

                loss = self.calc_val_loss(y_pred, y_true)  # .view(-1,1)
                val_loss = loss.item()
                running_loss += val_loss
                if self.verbosity >=1:
                    print("Epoch {}[{}/{}] Val Loss:{}".format(epoch, i, n_batch, val_loss))
        # if print_val_confusion and save_predictions:
        y_pred, y_true = np.hstack(Y['pred']), np.hstack(Y['true'])
        print("y_true: ")
        print(y_true.dtype)
        print("y_pred: " )
        print(y_pred.dtype)   
        print(classification_report(y_true, y_pred))
        running_loss /= n_batch
        return running_loss, f1_score(y_true, y_pred,average='macro')

    # @pysnooper.snoop("test_loop.log")
    def test_loop(self, test_dataloader):
        """Calculate final predictions on loss.

        Parameters
        ----------
        test_dataloader:DataLoader
                Test dataset.

        Returns
        -------
        array
                Predictions or embeddings.
        """
        # self.model.train(False) KEEP DROPOUT? and BATCH NORM??
        self.model.eval()
        y_pred = []
        Y_true = []
        running_loss = 0.
        n_batch = len(
            test_dataloader.dataset) // test_dataloader.batch_size
        print(str(n_batch))
        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(test_dataloader),total=n_batch):
                #X = Variable(batch[0],requires_grad=False)
                X, y_true = batch[:2]
                if len(batch)==3: Z=batch[2]
                else: Z=None
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_true = y_true.cuda()
                    if Z is not None: Z=Z.cuda()

                prediction = self.model(X) if Z is None else self.model(X,Z)
                #prediction=torch.sigmoid(prediction)[0]
                prediction=torch.sigmoid(prediction)
                #y_pred.append(prediction.detach().cpu().numpy())
                print(type(prediction))
                y_pred.extend(prediction.detach().cpu().numpy())
                print(type(prediction.detach().cpu().numpy()))
                Y_true.append(y_true.detach().cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)  # torch.cat(y_pred,0)
        y_true = np.concatenate(Y_true, axis=0).flatten()
        return y_pred,y_true

    def fit(self, train_dataloader, verbose=False, print_every=10, save_model=True, plot_training_curves=False, plot_save_file=None, print_val_confusion=True, save_val_predictions=True):
        """Fits the segmentation or classification model to the patches, saving the model with the lowest validation score.

        Parameters
        ----------
        train_dataloader:DataLoader
                Training dataset.
        verbose:bool
                Print training and validation loss?
        print_every:int
                Number of epochs until print?
        save_model:bool
                Whether to save model when reaching lowest validation loss.
        plot_training_curves:bool
                Plot training curves over epochs.
        plot_save_file:str
                File to save training curves.
        print_val_confusion:bool
                Print validation confusion matrix.
        save_val_predictions:bool
                Print validation results.

        Returns
        -------
        self
                Trainer.
        float
                Minimum val loss.
        int
                Best validation epoch with lowest loss.

        """
        # choose model with best f1
        self.train_losses = []
        self.val_losses = []
        self.val_f1 = []
        self.batch_val_losses = []
        self.batch_val_f1 = []
        if verbose:
            self.verbosity+=1
        for epoch in range(self.n_epoch):
            self.seed=self.initial_seed+epoch
            np.random.seed(self.seed)
            start_time = time.time()
            train_loss = self.train_loop(epoch, train_dataloader)
            current_time = time.time()
            train_time = current_time - start_time
            self.train_losses.append(train_loss)
            val_loss, val_f1 = self.val_loop(epoch, self.validation_dataloader,
                                     print_val_confusion=print_val_confusion, save_predictions=save_val_predictions)
            val_time = time.time() - current_time
            self.val_losses.append(val_loss)
            self.val_f1.append(val_f1)
            self.batch_val_losses.append(val_loss)
            self.batch_val_f1.append(val_f1)
            if True:#verbose and not (epoch % print_every):
                if plot_training_curves:
                    self.plot_train_val_curves(plot_save_file)
                print("Epoch {}: Train Loss {}, Val Loss {}, Train Time {}, Val Time {}".format(
                    epoch, train_loss, val_loss, train_time, val_time))
            self.save_best_val_model(val_loss, val_f1, self.val_losses, self.val_f1, epoch, save_model)
        if save_model:
            print("Saving best model at epoch {}".format(self.best_epoch))
            self.model.load_state_dict(self.best_model_state_dict)
        return self, self.min_val_loss_f1, self.best_epoch

    def save_best_val_model(self, val_loss, val_f1, val_loss_list, val_f1_list, epoch, save_model=True, batch=0):
        if (val_loss <= min(val_loss_list) if self.save_metric=='loss' else val_f1 >= max(val_f1_list)) and save_model:
            print("New best model at epoch {}".format(epoch))
            self.min_val_loss_f1 = val_loss if self.save_metric=='loss' else val_f1
            self.best_epoch = epoch
            if batch: self.best_batch = batch
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            self.save_checkpoint(self.best_model_state_dict,epoch,batch)

    def plot_train_val_curves(self, save_file=None):
        """Plots training and validation curves.

        Parameters
        ----------
        save_file:str
                File to save to.

        """
        plt.figure()
        sns.lineplot('epoch', 'value', hue='variable',
                     data=pd.DataFrame(np.vstack((np.arange(len(self.train_losses)), self.train_losses, self.val_losses)).T,
                                       columns=['epoch', 'train', 'val']).melt(id_vars=['epoch'], value_vars=['train', 'val']))
        if save_file is not None:
            plt.savefig(save_file, dpi=300)

    def predict(self, test_dataloader):
        """Make classification segmentation predictions on testing data.

        Parameters
        ----------
        test_dataloader:DataLoader
                Test data.

        Returns
        -------
        array
                Predictions.

        """
        y_pred,y_true = self.test_loop(test_dataloader)
        return y_pred,y_true

    def fit_predict(self, train_dataloader, test_dataloader):
        """Fit model to training data and make classification segmentation predictions on testing data.

        Parameters
        ----------
        train_dataloader:DataLoader
                Train data.
        test_dataloader:DataLoader
                Test data.

        Returns
        -------
        array
                Predictions.

        """
        return self.fit(train_dataloader)[0].predict(test_dataloader)

    def return_model(self):
        """Returns pytorch model.
        """
        return self.model


class CustomLoss(nn.Module):

    def __init__(self, weights=[1, 1, 1, 2, 1, 3]):
        # 'bce', 'mbce', 'dice', 'mse', 'msge', 'ddmse'
        super(CustomLoss, self).__init__()
        self.weights = np.array(weights)
#         self.weights = self.weights / sum(self.weights)
    
    @staticmethod
    def dice_loss(pred, gt, epsilon=1e-3):
        n = 2. * torch.sum(pred * gt)
        d = torch.sum(pred + gt)
        return 1. - (n + epsilon) / (d + epsilon)
    
    @staticmethod
    def get_gradient(maps):
        """
        Reference: some codes from https://github.com/vqdang/hover_net/blob/master/src/model/graph.py
        """
        def get_sobel_kernel(size):
            assert size % 2 == 1, 'Must be odd, get size={}'.format(size)

            h_range = np.arange(-size//2 + 1, size//2 + 1, dtype=np.float32)
            v_range = np.arange(-size//2 + 1, size//2 + 1, dtype=np.float32)
            h, v = np.meshgrid(h_range, v_range)
            kernel_h = h / (h * h + v * v + 1.0e-15)
            kernel_v = v / (h * h + v * v + 1.0e-15)
            return kernel_h, kernel_v 
        
        batchsize_ = maps.shape[0]
        hk, vk = get_sobel_kernel(5)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        hk = torch.tensor(hk, requires_grad=False).view(1, 1, 5, 5).to(device)
        vk = torch.tensor(vk, requires_grad=False).view(1, 1, 5, 5).to(device)

        h = maps[..., 0].unsqueeze(1)
        v = maps[..., 1].unsqueeze(1)

        dh = F.conv2d(h, hk, padding=2).permute(0, 2, 3, 1)
        dv = F.conv2d(v, vk, padding=2).permute(0, 2, 3, 1)
        return torch.cat((dh, dv), axis=-1)

    @staticmethod
    def dot_distance_loss(pred_dot, pred_hv, gt_dot, gt_hv):
        pred_dot = pred_dot >= 0.5
        pred_focus = torch.cat((pred_dot, pred_dot), axis=-1)
        gt_focus = torch.cat((gt_dot, gt_dot), axis=-1)
        pred_area = pred_focus * pred_hv
        gt_area = gt_focus * gt_hv
        return F.mse_loss(pred_area, gt_area)

    def msge_loss(self, pred, gt, focus):
        focus = torch.cat((focus, focus), axis=-1)
        pred_grad = self.get_gradient(pred)
        gt_grad = self.get_gradient(gt)
        # loss = pred_grad - gt_grad
        # loss = focus * (loss * loss)
        # loss = torch.sum(loss) / (torch.sum(loss) + 1.0e-8)
        return F.mse_loss(pred_grad, gt_grad)
        # return loss

    def forward(self, preds, gts, contain='single'):
        # transpose gts to channel last
        #gts = gts.permute(0, 2, 3, 1)
        gt_seg, gt_hv, gt_dot = torch.split(gts[..., :4], [1, 2, 1], dim=-1)
        pred_seg, pred_hv, pred_dot = torch.split(preds[..., :4], [1, 2, 1], dim=-1)
        # binary cross entropy loss
        bce = F.binary_cross_entropy(pred_seg, gt_seg)
        # masked binary cross entropy loss
        mbce = F.binary_cross_entropy(pred_dot * gt_dot, gt_dot) * 3 + F.binary_cross_entropy(pred_dot, gt_dot)
        # mbce = F.binary_cross_entropy(pred_dot, gt_dot)
        # dice loss
        dice = self.dice_loss(pred_seg, gt_seg)
        # mean square error of distance maps and their gradients
        mse = F.mse_loss(pred_hv, gt_hv)
        msge = self.msge_loss(pred_hv, gt_hv, gt_seg)
        # mean square error for dot and distance maps
        ddmse = self.dot_distance_loss(pred_dot, pred_hv, gt_dot, gt_hv)
        
        loss = bce * self.weights[0] + mbce * self.weights[1] + dice * self.weights[2] + mse * self.weights[3] + msge * self.weights[4] + ddmse * self.weights[5]

        if contain == 'single':
            return loss
        
        names = ('loss', 'bce', 'mbce', 'dice', 'mse', 'msge', 'ddmse')
        losses = [loss, bce, mbce, dice, mse, msge, ddmse]
        # if prefix is not None:
        #     names = ['{}_{}'.format(prefix, n) for n in names]
        return {name: loss for name, loss in zip(names, losses)}


class DiceLoss1(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss1, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE