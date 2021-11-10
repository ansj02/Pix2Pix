from dataIO import *
from models import *
from train_test import *

if __name__ == '__main__':
    epoch = 100
    k = 1
    l = 3
    lr = 2e-4
    drop_prob = 0.2
    batch_size = 4
    data_size = 400
    img_size = (256, 256)
    input_channels = 3
    output_channels = 3
    mode = 'test'  # train, test
    data_path = './pix2pix-dataset/facades/facades/train/'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)

    model = Model(input_channels, output_channels, batch_size).to(device)

    max_size = len(os.listdir(data_path))
    if data_size > max_size: data_size = max_size
    iteration = int(data_size / batch_size)


    if mode == 'train':
        opt_gen = torch.optim.AdamW(model.generator.parameters(), lr)
        opt_dis = torch.optim.AdamW(model.discriminator.parameters(), lr)
        real_img_set, cond_img_set = get_data_set(data_path, data_size, img_size)

        real_img_set = torch.tensor(real_img_set, dtype=torch.float32).to(device)
        cond_img_set = torch.tensor(cond_img_set, dtype=torch.float32).to(device)

        if os.path.isfile('model_data.pth'): model.load_state_dict(torch.load('model_data.pth'))

        for ep in range(epoch):
            print("=========  epoch : ", ep, "/", epoch, "  =========")
            for i in range(iteration):
                real_img_batch = real_img_set[i*batch_size:(i+1)*batch_size]
                cond_img_batch = cond_img_set[i*batch_size:(i+1)*batch_size]
                disc_loss, gen_loss = train(k, l, model, opt_gen, opt_dis, real_img_batch, cond_img_batch, device=device)
            print("disc_loss : %f, gen_loss : %f" % (disc_loss, gen_loss))
            torch.save(model.state_dict(), 'model_data.pth')
            print('model saved')

    elif mode == 'test':
        sample_size = 20
        model = Model(input_channels, output_channels, batch_size).to('cpu')

        real_img_set, cond_img_set = get_data_set(data_path, sample_size, img_size)
        if os.path.isfile('model_data.pth'): model.load_state_dict(torch.load('model_data.pth'))
        make_sample_img(model, cond_img_set, real_img_set, sample_size)

