def train(k, l, model, optimizer_gen, optimizer_dis, batch_img, batch_cond_img, device='cpu'):
    #disc_loss, gen_loss = model.loss(batch_img, batch_cond_img)
    for _ in range(k):
        disc_loss = model.loss_dis(batch_img, batch_cond_img)
        optimizer_dis.zero_grad()
        disc_loss.backward()
        optimizer_dis.step()
    for _ in range(l):
        gen_loss = model.loss_gen(batch_img, batch_cond_img)
        optimizer_gen.zero_grad()
        gen_loss.backward()
        optimizer_gen.step()
    return disc_loss, gen_loss