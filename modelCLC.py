###################
##### IMPORTS #####
###################
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
#################
#### DATASET ####
#################
class PhishingURLDataset(Dataset):
    '''define Pytorch-Friendly Dataset'''
    def __init__(self, urls, labels, char_to_int_map, embedding_dim, max_len=128):
        self.urls = urls #url data
        self.labels = labels #label data
        self.char_to_int_map = char_to_int_map  #mapping generated earlier 
        self.embedding_dim = embedding_dim #hyperparameter common to cnn & ds
        self.max_len = max_len
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        
        #convert URL to integer encoding based on char map
        url_encoded = torch.zeros(self.max_len, dtype=torch.long)  #adjust to integer type for embedding layer
        for i, char in enumerate(url[:self.max_len]):
            if char in self.char_to_int_map:
                url_encoded[i] = self.char_to_int_map[char]
            else:
                url_encoded[i] = 1  #1 by definition => unknown char (not on map); chars that arent covered here will be left as 0 (padding)
        
        return url_encoded, torch.tensor(label, dtype=torch.float)

###############
#### MODEL ####
###############

class URLPhishingClassifier(nn.Module):
    '''model definition with (trainable) embedding layer for character-level embeddings'''
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, hidden_dim, num_classes):
        super(URLPhishingClassifier, self).__init__()
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #conv layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        
        #dnn
        self.fc1 = nn.Linear(num_filters * len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        #final
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)     #channel dimension for CNN
        convs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pools = [torch.max(conv, dim=2)[0] for conv in convs]
        out = torch.cat(pools, dim=1)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return self.sigmoid(out).squeeze(1)
    
#############################################

from helperDS import clean_url #URL prefix cleaning

#############################################################################################  MY_CLC  #############################################################################################

class MyCLC:
    def __init__(self, df, df_val = None, prep = False, prefix = True, to_lower=True, split = True, plots = True, embedding_dim = 200, num_filters = 256, filter_sizes = [3,7,8], hidden_dim = 512, num_classes = 1, batch_size = 256*4, learning_rate = 5e-4, num_epochs = 10, gpu_parallel=False):
        self.split = split
        self.prep = prep
        self.prefix = prefix
        self.to_lower = to_lower
        self.plots = plots
        self.gpu_parallel=gpu_parallel
        
        self.df = df
        self.df_val = df_val
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


        self.num_classes = num_classes # always 1
        if num_classes!=1:
            raise Exception(f"num_classes should be 1 but is {num_classes}.")
        
        self.cmap = None
        self.vocab_size = None
 
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

        self.model = None
        self.device = None

        self.test_accs = []
        self.val_accs = []
        self.epochs_val = []

    def preprocess_df(self, df):
        '''preprocess function -> takes care of preprocessing #discarder implemented after'''
        if self.prefix:
            df['url'] = df['url'].apply(clean_url) #double clean
            df['url'] = df['url'].apply(clean_url)
        if self.to_lower:
            df['url'] = df['url'].str.lower()
        return df

    def generate_cmap(self, df_t):
        char_counts = Counter(char for url in df_t['url'] for char in url)
        frequency_threshold = int(len(df_t) / 1600)
        print("frequency_threshold =",frequency_threshold)
        relevant_chars = {char for char, count in char_counts.items() if count >= frequency_threshold}

        random.seed(42)
        unique_integers = random.sample(range(2, len(relevant_chars) + 2), len(relevant_chars)) #1 will be left for 'UNK' in vocab(size) and 0 for padding by default
        self.cmap = dict(zip(relevant_chars, unique_integers))
        self.rev_cmap = {v: k for k, v in self.cmap.items()}
        self.vocab_size = len(self.cmap) + 2
        print("cmap =",self.cmap)

    #############################################################################

    def loading(self):
        '''this function loads dataset and preprocesses dfs given if prep = True'''
        
        #preprocessing (if prep not done outside, set prep = True)
        if self.prep == True:
            print("Intiailising preprocessing.")
            self.df = self.preprocess_df(self.df)
            print("Preprocessing for main dataframe finished.")
            if self.df_val is not None:
                self.df_val = self.preprocess_df(self.df_val)
                print("Preprocessing for validation dataframe finished.")
            print("")
        
        split = self.split
        if split == False:
            urls_train = self.df['url']
            labels_train = self.df['status']
        else:
            testSize = 0.2 ##hardcoded
            print(f"Initialising {100*(1-testSize)}/{100*testSize} % train/test splitting.")
            df_train, df_test = train_test_split(self.df, test_size=testSize, random_state=42)
            df_train = df_train.reset_index(drop = True) #important to resetindex, otherwise we probably wont have 0 or 1 and the for loop will break in train
            df_test = df_test.reset_index(drop = True)

            urls_train = df_train['url']
            urls_test = df_test['url']

            labels_train = df_train['status']
            labels_test = df_test['status']
            print("Finished splitting.")
            print("")

        if self.df_val is not None:
            df_val = self.df_val.reset_index(drop = True) #also important to reset index
            urls_val = df_val['url']
            labels_val = df_val['status']

        print("Generating character map.")
        if split: self.generate_cmap(df_train)
        else: self.generate_cmap(self.df) ##charmap should be in a char_map + 2 (padding and UNK, ideally masks for |ID> and others)
        print("Finished generating charmap.")
        print("")

        #To-Do: self.generate_charLenMax for phishing dataset (defaulted 128), but we can change it based on training data (ex. 90% cumulative)
        
        #init model and dataloaders
        print("Starting data loaders.")
        count_dataLoaders = 0
        self.model = URLPhishingClassifier(self.vocab_size, self.embedding_dim, self.num_filters, self.filter_sizes, self.hidden_dim, self.num_classes)

        if self.gpu_parallel:
            self.model = nn.DataParallel(self.model)

        #always load the train
        train_dataset = PhishingURLDataset(urls_train, labels_train, self.cmap, self.embedding_dim)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        count_dataLoaders+=1

        if split == True: #if split, we also load the test
            test_dataset = PhishingURLDataset(urls_test, labels_test, self.cmap, self.embedding_dim)
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            count_dataLoaders+=1
        
        if self.df_val is not None:
            val_dataset = PhishingURLDataset(urls_val, labels_val, self.cmap, self.embedding_dim)
            self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            count_dataLoaders+=1
        print(f"Sucessfully loaded {count_dataLoaders} Data Loaders.")
        print("\n\n-----------------------------------------------------------------------------------------\n\n")

    ##########################################  plotting  ##########################################

    def plot_metrics(self):
        epochs = self.epochs_val

        if len(self.val_accs)!=0: #test and val
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, self.test_accs, label='Test Accuracy', color='red', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            plt.title('Test Accuracy over Epochs')
            plt.grid(True)
            plt.legend()

            #accuracy
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.val_accs, label='Validation Accuracy', color='blue', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Val. Accuracy')
            plt.title('Validation Accuracy over Epochs')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()
            
        else:
            if self.split:
                plt.figure(figsize=(12, 5))
                plt.plot(epochs, self.test_accs, label='Test Accuracy', color='red', marker='o')
                plt.xlabel('Epochs')
                plt.ylabel('Test Accuracy')
                plt.title('Test Accuracy over Epochs')
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.show()
            else: print("No plots to show, no train-test split.")

    ###################################### training / testing ######################################

    def train_and_eval(self):
        '''performs training and evaluation
        if nr.epochs >= 10 we do an evaluation for each 10% step
        evaluation: test and validation sets [use any or both, whats available]
        '''

        print("Device " + str(self.device) + " is available. Initialising training:")

        #move to GPU
        self.model = self.model.to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #training loop
        for epoch in range(self.num_epochs):
            self.model.train()  #model to training mode
            epoch_loss = 0  #init epoch loss

            #init tqdm progress bar
            progress_bar = tqdm(enumerate(self.train_dataloader), 
                                total=len(self.train_dataloader), 
                                desc=f"Epoch [{epoch+1}/{self.num_epochs}]")

            for batch_idx, (urls, labels) in progress_bar:
                urls, labels = urls.to(self.device), labels.to(self.device)  #move everything to gpu

                outputs = self.model(urls)  #forward pass
                loss = criterion(outputs, labels)  #compute loss

                optimizer.zero_grad()  #zero the gradient buffer
                loss.backward()  #backpropagation
                optimizer.step()  #perform a single optimization step

                epoch_loss += loss.item()  #accumulate loss

                #update tqdm progress bar with batch index and loss
                progress_bar.set_postfix(batch_idx=batch_idx + 1, loss=loss.item())

            print(f"Epoch [{epoch+1}/{self.num_epochs}] completed with average loss: {epoch_loss / len(self.train_dataloader):.4f}")  #print epoch summary
            # if 10% epochs done, 20% epochs done, ... [implies we test set or validate set]
            if (self.num_epochs>=10 and epoch % (self.num_epochs // 10) == 0 or (epoch == self.num_epochs-1)) or self.num_epochs < 10: #if more than 10 epochs, do 1 eval per 10%. if not, do every time
                print("_____________________________________________________________\n")
                self.epochs_val.append(epoch+1)
                if self.val_dataloader is None:
                    if self.split == True:
                        self.test() #else nothing
                else:
                    if self.split == True:
                        self.test()
                        print("_____________________________________________________________")
                        self.test(val = True)
                    else:
                        self.test(val = True)
            if epoch!=self.num_epochs-1:
                print("\n\n--------------------------------------------------------------------------------------------\n\n")
        print("")
        
        ##end of training, print end
        if self.val_dataloader is not None:
            if self.split == True:
                print("Finished. Training, Testing and Validation Complete!") 
            else:
                print("Finished. Training and Validation Complete!")
        else:
            if self.split == True:
                print("Finished. Training and Testing Complete!") 
            else:
                print("Training Complete!")

    ############################################################################

    def test(self, df_out=None, val = False, return_metrics = False):
        '''multi-purpose test; either with initialised val, or outside/external [ex. val hispar, external OP, PT]
           note: expects 0-based indexing [.reset_index(drop = True)]
        '''
        print("Device "+str(self.device)+" is available. Initialising evaluation:")

        self.model.eval()

        labels_obtained_by_model = []
        true_labels = []

        if df_out is not None and not df_out.empty: #if outside df given, use it
            urls_out = df_out['url']
            labels_out = df_out['status']
            out_dataset = PhishingURLDataset(urls_out, labels_out, self.cmap, self.embedding_dim)
            outside_dataloader = DataLoader(out_dataset, batch_size=self.batch_size, shuffle=False) #no need to save in self

            progress_bar = tqdm(enumerate(outside_dataloader), 
            total=len(outside_dataloader), 
            desc=f"External Dataset")

        else: #otherwise validate or test (if nothing return)
            if val == False:
                if self.split==True:
                    progress_bar = tqdm(enumerate(self.test_dataloader), 
                    total=len(self.test_dataloader), 
                    desc=f"Test Set (80-20 split)")
                else:
                    print("No Test, Validation, or External Set given.")
                    return
            else:
                progress_bar = tqdm(enumerate(self.val_dataloader), 
                total=len(self.val_dataloader), 
                desc=f"Validation Set")

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for b_idx, (urls, labels) in progress_bar:
                urls = urls.to(self.device)  #move data to gpu
                labels = labels.to(self.device) 

                outputs = self.model(urls)

                #convert outputs to binary predictions (0/1)
                predicted_labels = (outputs > 0.5).int()  #threshold for classification
                
                #append preds and true labels to lists
                labels_obtained_by_model.extend(predicted_labels.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

                ##progress bar
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)
                running_accuracy = np.around(total_correct / total_samples, 3)
                
                progress_bar.set_postfix(batch_idx=b_idx + 1, accuracy = running_accuracy)

        #to numpy arrays for metric computation
        labels_obtained_by_model = np.array(labels_obtained_by_model)
        true_labels = np.array(true_labels)

        acc = np.around(accuracy_score(true_labels, labels_obtained_by_model),3)
        prec = np.around(precision_score(true_labels, labels_obtained_by_model),3)
        rec = np.around(recall_score(true_labels, labels_obtained_by_model),3)
        f1 = np.around(f1_score(true_labels, labels_obtained_by_model),3)
        print("Accuracy:",acc)
        print("Precision:",prec)
        print("Recall:",rec)
        print("F1-Score:",f1)

        #save for plots
        if val == True:
            self.val_accs.append(acc)
        else:
            self.test_accs.append(acc) #add exception, if no test available earlier 
        
        if return_metrics==True: return [acc, prec, rec, f1]

    ################################################################################################
    
    def run(self): #main
        '''main pipeline of CLC'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loading()
        self.train_and_eval()
        if self.plots: self.plot_metrics()

    ###################################### saving and loading ######################################
    def save_model(self, file_path):
        """Saves the model's state dictionary along with hyperparameters."""
        if self.model is None:
            raise ValueError("Model is not initialized. Train or load a model before saving.")
        
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_filters": self.num_filters,
            "filter_sizes": self.filter_sizes,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "char_map": self.cmap,  #save character map for reproducibility (very important, as if ran again order will be)
        }
        torch.save(save_data, file_path)
        print(f"Model and parameters saved to {file_path}")
    
    def load_model(self, file_path):
        """Loads the model's state dictionary and hyperparameters."""
        checkpoint = torch.load(file_path)
        
        #extract parameters
        self.vocab_size = checkpoint["vocab_size"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.num_filters = checkpoint["num_filters"]
        self.filter_sizes = checkpoint["filter_sizes"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_classes = checkpoint["num_classes"]
        self.cmap = checkpoint["char_map"]
        
        #re-initialize the model with loaded parameters
        self.model = URLPhishingClassifier(
            vocab_size=self.vocab_size, 
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters, 
            filter_sizes=self.filter_sizes, 
            hidden_dim=self.hidden_dim, 
            num_classes=self.num_classes,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(f"We just set device to {self.device}.")
        self.model = self.model.to(self.device)
        print(f"Model and parameters loaded from {file_path}")