import utility
import torch
from torch.autograd import grad


def s_test(z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0, repeat=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        repeat: int, number of iterations aka recursion depth

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model, gpu)
    h_init_estimates = v.copy()

    for _ in utility.create_progressbar(repeat, desc='s_test'):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once
        for x, t in z_loader:
            if gpu >= 0:
                x, t = x.cuda(gpu), t.cuda(gpu)
            y = model(x)
            loss = model.calc_loss(y, t)
            hv = hvp(loss, list(model.parameters()), h_init_estimates)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * h_estimate - _hv / scale
                for _v, h_estimate, _hv in zip(v, h_init_estimates, hv)]
            # h_estimate = []
            # for _v, h_estimate, _hv in zip(v, h_init_estimates, hv):
            #     h_estimate = (_v + (1 - damp) * h_estimate - _hv / scale)
            break
    return h_estimate


def grad_z(z, t, model, gpu=-1):
    """Calculates the gradient

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    if gpu >= 0:
        z, t = z.cuda(gpu), t.cuda(gpu)
        model.cuda(gpu)
    y = model(z)
    loss = model.calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    return list(grad(loss, list(model.parameters()), create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


if __name__ == '__main__':
    from trainer_mnist import MnistTrainer
    from net import Net
    from optimizers import MomentumSGD
    model = Net()
    optimizer = MomentumSGD(model, 0, 0)
    main = MnistTrainer(model, optimizer, train_batch_size=1)
    z_test, t_test = main.test_loader.dataset[0]
    z_test = main.test_loader.collate_fn([z_test])
    t_test = main.test_loader.collate_fn([t_test])
    test = s_test(z_test, t_test, model, main.train_loader, gpu=-1)
