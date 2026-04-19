learning_rates = [2.9e-3, 3e-3 , 3.1e-3]
betas = [0.05, 0.1, 0.08]

best_loss = float('inf') 
best_params = {'lr': None, 'beta': None}
best_model = None

print("Starting Hyperparameter Grid Search...")
total = 9
i = 0
# 2. Loop through every combination
for lr in learning_rates:
    for b in betas:
        print(f"\n--- Testing LR: {lr} | Beta: {b} ---")
        
        model = VAE.Module(in_channels=3, latent_dim=128).to(device)
        
        avg_loss = model.startTraining(
            train_loader=train_loader, 
            ds_length=ds_length, 
            learning_rate=lr, 
            beta=b, 
            device=device, 
            epochs=10
        )
        
        # 3. Check if this combination is the new champion
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params['lr'] = lr
            best_params['beta'] = b
            best_model = model
            print(f"🌟 New Best Found! Loss: {best_loss:.4f}")
        i+=1
        print(f"{i} out of {total} cobinations completed")

print("\n=========================================")
print(f"🏆 Grid Search Complete!")
print(f"Best Learning Rate: {best_params['lr']:.2e}")
print(f"Best Beta: {best_params['beta']}")
print(f"Lowest Loss: {best_loss:.4f}")
print("=========================================")

import matplotlib.pyplot as plt

def generate_new_images(model, num_images=10, latent_dim=128, device='cpu'):
    #  ALWAYS put the model in evaluation mode before generating
    model.eval()
    
    # Tell PyTorch we don't need to track gradients (saves memory & runs faster)
    with torch.no_grad():
        # Create the random noise! 
        z = torch.randn(num_images, latent_dim).to(device)
        
        # Pass the noise through the decode method
        fake_images = model.decode(z)
        
        # Move images back to CPU for plotting
        fake_images = fake_images.cpu()

    # Plot the results
    fig = plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)

        img_tensor = fake_images[i]
        
        # Rearrange dimensions for Matplotlib (Shape [H, W, 3])
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        plt.imshow(img_np.clip(0, 1), interpolation='none') 
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

generate_new_images(best_model, num_images=10, latent_dim=128, device=device)