from dcgan import *

# Loss function
loss_function = nn.BCELoss().to(device)

G = Generator()
D = Discriminator()
earlystopping = EarlyStopping(patience=1000, verbose=1)

if is_cuda:
    G.cuda()
    D.cuda()

# Initialize weights
G.apply(weights_init_normal)
D.apply(weights_init_normal)

# path = 'E:/지정맥/real_frame'
train_path = '/workspace/GAN/real_frame'
# save_path = 'E:/지정맥/model/'
valid_path = '/workspace/GAN/valid'
save_path = './GAN/save_dcgan/'
save_img = './GAN/dcgan_img/'
test_path = '/workspace/GAN/test'

dc_trainp = 'E:/연구실/지정맥/GAN/dcgan_img_valid'
dc_testp = 'E:/연구실/지정맥/GAN/dcgan_img_test'
train_dataset = GANdata(dc_trainp)
# valid_dataset = GANdata(valid_path)
test_dataset = GANdata(dc_testp)

trainset = DataLoader(train_dataset,
                      shuffle=True,
                      batch_size=batch_size)

# validset = DataLoader(valid_dataset,
#                       shuffle=True,
#                       batch_size=batch_size)

testset = DataLoader(test_dataset,
                     shuffle=True,
                     batch_size=batch_size)

optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001)

for epoch in range(epochs):
    for i, image in enumerate(trainset):
        print('i', i)
        print('image.shape', image.shape)
        # image = image.reshape(batch_size, -1).to(device)
        print('re_image', image.shape)
        image = image.to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        print('real_lables', real_labels.shape)
        # D가 진짜 이미지를 진짜로 인식하는 오차
        outputs = D(image)
        print('outputs', outputs.shape)
        D_loss_real = loss_function(outputs, real_labels)
        real = outputs

        # G 동작 정의 (무작위 텐서로 fake 지정맥 이미지 생성)
        z = torch.randn(batch_size, noise).to(device)
        # z = torch.rnadn(image.size(0), )
        fake_image = G(z)

        # D가 가짜 이미지를 가짜로 인식하는 오차
        outputs = D(fake_image)
        D_loss_fake = loss_function(outputs, fake_labels)
        fake = outputs
        D_loss = D_loss_real + D_loss_fake

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # G가 D를 속였는지에 대한 오차
        fake_image = G(z)
        outputs = D(fake_image)
        G_loss = loss_function(outputs, real_labels)

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        G_loss.backward()
        optimizer_G.step()

        print('epoch [{}/{}] d_loss:{:.4f} g_loss: {:.4f} D(x):{:.2f} D(G(z)): {:.2f} '
              .format(epoch, epochs, D_loss.item(), G_loss.item(), real.mean().item(), fake.mean().item()))

    G.eval()
    D.eval()
    with torch.no_grad():
        for i, val_img in enumerate(testset):
            val_img = val_img.to(device)
            z = torch.randn(batch_size, noise).to(device)
            # val_img = val_img.reshape(batch_size, -1).to(device)
            val_real_labels = torch.ones(batch_size, 1).to(device)
            val_fake_labels = torch.zeros(batch_size, 1).to(device)

            val_fake_img = G(z)
            val_r_outputs = D(val_img).to(device)
            val_f_outputs = D(val_fake_img).to(device)

            val_fake_img = torch.reshape(val_fake_img, (100, 3, 64, 64)).to(device)
            print('shape', val_fake_img.shape)

            # G가 D를 속였는지에 대한 오차
            val_G_loss = loss_function(val_f_outputs, val_real_labels).to(device)
            val_D_loss = loss_function(val_r_outputs, real_labels) + loss_function(val_f_outputs, val_fake_labels).to(
                device)

            # val_img = val_img.reshape(batch_size, 3*80*60).to(device)

            print('epoch [{} / {}] val_d_loss{:.4f} val_g_loss {:.4f}'.format(epoch, epochs, val_D_loss.item(),
                                                                              val_G_loss.item()))

            # save_image(val_fake_img, '/workspace/GAN/fake_img/' + str(i) + '.jpg')
            # save_image(val_fake_img, 'workspace/GAN/dcgan_image/' + str(epoch) + str(i) + '.jpg')
            save_image(val_fake_img, "dcgan_img/%d.jpg" % epoch)

        # 모델 저장
        torch.save(G.state_dict(), "save_dcgan/%dgenerator.pt" % epoch)
        torch.save(D.state_dict(), "save_dcgan/%ddiscriminator.pt" % epoch)
        # torch.save(D.state_dict(),save_path+"discriminator" + str(val_G_loss) + ".pt")

    if earlystopping.validate(val_G_loss):
        break
