# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

import torch
from torch import nn

from dataloader import mnist_usps,mnist_Rmnist
from module import Encoder, AdversarialNetwork, DFC, adv_loss
from eval import predict, cluster_accuracy, balance
from utils import set_seed, AverageMeter, target_distribution, aff, inv_lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=512)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--iters", type=int, default=20000)
parser.add_argument("--test_interval", type=int, default=5000)
parser.add_argument("--adv_mult", type=float, default=10.0)
parser.add_argument("--coeff_fair", type=float, default=1.0)
parser.add_argument("--coeff_par", type=float, default=1.0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=2019)
parser.add_argument("--corrupted",type=float, default=0)
parser.add_argument("--corrupted_set",type=int, default=0)
parser.add_argument("--dataset",type=str, default="usps")
args = parser.parse_args()


def main():
    set_seed(args.seed)
    args.gpu = "cuda:0"
    torch.cuda.set_device(args.gpu)

    encoder = Encoder().cuda()
    encoder_group_0 = Encoder().cuda()
    encoder_group_1 = Encoder().cuda()

    dfc = DFC(cluster_number=args.k, hidden_dimension=64).cuda()
    dfc_group_0 = DFC(cluster_number=args.k, hidden_dimension=64).cuda()
    dfc_group_1 = DFC(cluster_number=args.k, hidden_dimension=64).cuda()

    critic = AdversarialNetwork(in_feature=args.k,
                                hidden_size=32,
                                max_iter=args.iters,
                                lr_mult=args.adv_mult).cuda()

    # encoder pre-trained with self-reconstruction
    encoder.load_state_dict(torch.load("./save/encoder_pretrain.pth"))

    # encoder and clustering model trained by DEC
    encoder_group_0.load_state_dict(torch.load("./save/encoder_mnist.pth"))
    dfc_group_0.load_state_dict(torch.load("./save/dec_mnist.pth"))
    #dataset dependend variables
    if args.dataset == "usps":
        encoder_group_1.load_state_dict(torch.load("./save/encoder_usps.pth"))
        dfc_group_1.load_state_dict(torch.load("./save/dec_usps.pth"))
        centers = np.loadtxt("./save/centers.txt")
        #data loader
        data_loader = mnist_usps(args)
        
    elif args.dataset == "Rmnist":
        encoder_group_1.load_state_dict(torch.load("./save/encoder_Rmnist.pth"))
        centers = np.loadtxt("./save/centers_Rmnist.txt")
        #data loader
        data_loader = mnist_Rmnist(args)
        
    #load clustering centroids given by k-means    
    cluster_centers = torch.tensor(centers, dtype=torch.float, requires_grad=True).cuda()
    with torch.no_grad():
        print("loading clustering centers...")
        dfc.state_dict()['assignment.cluster_centers'].copy_(cluster_centers)

    optimizer = torch.optim.Adam(dfc.get_parameters() + encoder.get_parameters() + critic.get_parameters(),
                                 lr=args.lr,
                                 weight_decay=5e-4)
                                 
    criterion_c = nn.KLDivLoss(reduction="sum")
    criterion_p = nn.MSELoss(reduction="sum")
    C_LOSS = AverageMeter()
    F_LOSS = AverageMeter()
    P_LOSS = AverageMeter()

    encoder_group_0.eval(), encoder_group_1.eval()
    dfc_group_0.eval(), dfc_group_1.eval()

    
    len_image_0 = len(data_loader[0])
    len_image_1 = len(data_loader[1])
    print(len_image_0,args.iters, args.iters/len_image_0)
    
    acc_list = []
    nmi_list = []
    bal_list = []
    en0_list = []
    en1_list = []
    for step in range(args.iters):
        encoder.train()
        dfc.train()
        if step % len_image_0 == 0:
            iter_image_0 = iter(data_loader[0])
        if step % len_image_1 == 0:
            iter_image_1 = iter(data_loader[1])

        image_0, _ = iter_image_0.__next__()
        image_1, _ = iter_image_1.__next__()
        image_0, image_1 = image_0.cuda(), image_1.cuda()
        image = torch.cat((image_0, image_1), dim=0)
        predict_0, predict_1 = dfc_group_0(encoder_group_0(image_0)[0]), dfc_group_1(encoder_group_1(image_1)[0])

        z, _, _ = encoder(image)
        output = dfc(z)
        output_0, output_1 = output[0:args.bs, :], output[args.bs:args.bs * 2, :]
        target_0, target_1 = target_distribution(output_0).detach(), target_distribution(output_1).detach()

        clustering_loss = 0.5 * criterion_c(output_0.log(), target_0) + 0.5 * criterion_c(output_1.log(), target_1)
        fair_loss = adv_loss(output, critic)
        partition_loss = 0.5 * criterion_p(aff(output_0), aff(predict_0).detach()) \
                         + 0.5 * criterion_p(aff(output_1), aff(predict_1).detach())
        total_loss = clustering_loss + args.coeff_fair * fair_loss + args.coeff_par * partition_loss

        optimizer = inv_lr_scheduler(optimizer, args.lr, step, args.iters)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        C_LOSS.update(clustering_loss)
        F_LOSS.update(fair_loss)
        P_LOSS.update(partition_loss)
        if step % 100 == 0:
            print(step)
        if step % args.test_interval == args.test_interval - 1 or step == 0:
            if args.corrupted != 0 :
                torch.save(critic.state_dict(), "./save/critic_Cor"+str(args.corrupted)+"_"+args.dataset + "_" + args.corrupted+str(step)+".pth")
                torch.save(dfc.state_dict(), "./save/dfc_Cor"+str(args.corrupted)+"_"+args.dataset+"_" + args.corrupted+ str(step)+".pth")
                torch.save(encoder.state_dict(), "./save/encoder_Cor"+str(args.corrupted)+"_"+args.dataset+"_" + args.corrupted+str(step)+".pth")
            else:
                torch.save(critic.state_dict(), "./save/critic_"+args.dataset + str(step)+".pth")
                torch.save(dfc.state_dict(), "./save/dfc_"+args.dataset +str(step)+".pth")
                torch.save(encoder.state_dict(), "./save/encoder_"+args.dataset"+str(step)+".pth")
            predicted, labels = predict(data_loader, encoder, dfc)
            predicted, labels = predicted.cpu().numpy(), labels.numpy()
            _, accuracy = cluster_accuracy(predicted, labels, 10)
            nmi = normalized_mutual_info_score(labels, predicted, average_method="arithmetic")
            bal, en_0, en_1 = balance(predicted, 60000)
            print("Step:[{:03d}/{:03d}]  "
                  "Acc:{:2.3f};"
                  "NMI:{:1.3f};"
                  "Bal:{:1.3f};"
                  "En:{:1.3f}/{:1.3f};"
                  "C.loss:{C_Loss.avg:3.2f};"
                  "F.loss:{F_Loss.avg:3.2f};"
                  "P.loss:{P_Loss.avg:3.2f};".format(step + 1, args.iters, accuracy, nmi, bal, en_0,
                                                     en_1, C_Loss=C_LOSS, F_Loss=F_LOSS, P_Loss=P_LOSS))
            acc_list += [str(accuracy) + " "]
            nmi_list += [str(nmi)+ " "]
            bal_list += [str(bal) + " "]
            en0_list += [str(en_0)+ " "]
            en1_list += [str(en_1)+ " "]
        
    file = open("results_new.txt", "a")
    file.writelines(acc_list)
    file.write('\n')
    file.writelines(nmi_list)
    file.write('\n')
    file.writelines(bal_list)
    file.write('\n')
    file.writelines(en0_list)
    file.write('\n')
    file.writelines(en1_list)
    return


if __name__ == "__main__":
    main()
