from abc import ABC, abstractmethod
import tensorflow as tf

from codebase.mlp import MLP

# defaults
EPS = 1e-8
HIDDEN_LAYER_SPECS = {  # format as dict that maps network to list of hidden layer widths
    'enc': [],
    'cla': [],
    'rec': [],
    'aud': [5],
}
CLASS_COEFF = 1.
FAIR_COEFF = 0.
RECON_COEFF = 0.
XDIM = 61
YDIM = 1
ZDIM = 10
ADIM = 1
A_WTS = [1., 1.]
Y_WTS = [1., 1.]
AY_WTS = [[1., 1.], [1., 1.]]
SEED = 0
ACTIV = 'leakyrelu'
HINGE = 0.


class AbstractBaseNet(ABC):
    def __init__(self,
                 recon_coeff=RECON_COEFF,
                 class_coeff=CLASS_COEFF,
                 fair_coeff=FAIR_COEFF,
                 xdim=XDIM,
                 ydim=YDIM,
                 zdim=ZDIM,
                 adim=ADIM,
                 hidden_layer_specs=HIDDEN_LAYER_SPECS,
                 seed=SEED,
                 hinge=HINGE,
                 **kwargs):
        self.recon_coeff = recon_coeff
        self.class_coeff = class_coeff
        self.fair_coeff = fair_coeff
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.adim = adim
        self.hidden_layer_specs = hidden_layer_specs
        self.seed = seed
        self.hinge = hinge
        tf.set_random_seed(self.seed)
        self._define_vars()
        self.Z = self._get_latents(self.X)
        self.Y_hat_logits = self._get_class_logits(self.Z)
        self.Y_hat = self._get_class_preds_from_logits(self.Y_hat_logits)
        self.A_hat_logits = self._get_sensitive_logits(self._get_aud_inputs())
        self.A_hat = self._get_aud_preds_from_logits(self.A_hat_logits)
        self.X_hat = self._get_recon_inputs(self.Z)
        self.class_loss = self._get_class_loss(self.Y_hat, self.Y)
        self.recon_loss = self._get_recon_loss(self.X_hat, self.X)
        self.aud_loss = self._get_aud_loss(self.A_hat, self.A)
        self.loss = self._get_loss()
        self.class_err = classification_error(self.Y, self.Y_hat)
        self.aud_err = classification_error(self.A, self.A_hat)

    def _get_aud_inputs(self):
        return self.Z

    @abstractmethod
    def _define_vars(self):  # declare tensorflow variables and placeholders
        pass

    @abstractmethod
    def _get_latents(self, inputs, scope_name='model/enc_cla', *args):  # map inputs to latents
        pass

    @abstractmethod
    def _get_class_logits(self, latents, scope_name='model/enc_cla'):  # map latents to class logits
        pass

    @abstractmethod
    def _get_sensitive_logits(self, latents, scope_name='model/aud', *args):  # map latents to sensitive logits (i.e., the adversary/auditor)
        pass

    @abstractmethod
    def _get_recon_inputs(self, latents, scope_name='model/enc_cla'):  # reconstruct inputs from latents
        pass

    @abstractmethod
    def _get_class_loss(self, pred, target):  # produce losses for the classification task
        pass

    @abstractmethod
    def _get_recon_loss(self, pred, target):  # produce losses for the reconstruction task
        pass

    @abstractmethod
    def _get_aud_loss(self, pred, target, *args):  # produce losses for the fairness task
        pass

    @abstractmethod
    def _get_loss(self):  # produce losses for the fairness task
        pass

    @abstractmethod
    def _get_class_preds_from_logits(self, logits):  # map inputs to test-time evaluation scores
        pass

    @abstractmethod
    def _get_aud_preds_from_logits(self, logits):  # map inputs to test-time evaluation scores
        pass


class DemParGan(AbstractBaseNet):
    def _define_vars(self):
        assert(
            isinstance(self.hidden_layer_specs, dict) and
            all([net_name in self.hidden_layer_specs for net_name in ['enc', 'cla', 'aud', 'rec']])
        )
        self.X = tf.placeholder("float", [None, self.xdim], name='X')
        self.Y = tf.placeholder("float", [None, self.ydim], name='Y')
        self.A = tf.placeholder("float", [None, self.adim], name='A')
        self.epoch = tf.placeholder("float", [1], name='epoch')
        return

    def _get_latents(self, inputs, scope_name='model/enc_cla', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='inputs_to_latents',
                      shapes=[self.xdim] + self.hidden_layer_specs['enc'] + [self.zdim],
                      activ=ACTIV)
            return mlp.forward(inputs)

    def _get_class_logits(self, latents, scope_name='model/enc_cla', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='latents_to_class_logits',
                      shapes=[self.zdim] + self.hidden_layer_specs['cla'] + [self.ydim],
                      activ=ACTIV)
            return mlp.forward(latents)

    def _get_sensitive_logits(self, latents, scope_name='model/aud', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='latents_to_sensitive_logits',
                      shapes=[self.zdim] + self.hidden_layer_specs['aud'] + [self.adim],
                      activ=ACTIV)
            return mlp.forward(latents)

    def _get_recon_inputs(self, latents, scope_name='model/enc_cla'):
        with tf.variable_scope(scope_name):
            mlp = MLP(name='latents_to_reconstructed_inputs',
                      shapes=[self.zdim + 1] + self.hidden_layer_specs['rec'] + [self.xdim],
                      activ=ACTIV)
            Z_and_A = tf.concat([self.Z, self.A], axis=1)
            final_reps = mlp.forward(Z_and_A)
            return final_reps

    def _get_class_loss(self, Y_hat, Y):
        return cross_entropy(Y, Y_hat)

    def _get_recon_loss(self, X_hat, X):
        return tf.reduce_mean(tf.square(X - X_hat), axis=1)

    def _get_aud_loss(self, A_hat, A):
        return cross_entropy(A, A_hat)

    def _get_weight_decay(self):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/aud')
        weights_norm = [tf.reduce_sum(tf.square(w)) for w in var_list]
        weight_norm_tensor = tf.stack(weights_norm)
        return tf.reduce_sum(weight_norm_tensor)

    def _get_loss(self):  # produce losses for the fairness task
        return tf.reduce_mean([
            self.class_coeff*self.class_loss,
            self.recon_coeff*self.recon_loss,
            -self.fair_coeff*self.aud_loss
        ])

    def _get_class_preds_from_logits(self, logits):
        return tf.nn.sigmoid(logits)

    def _get_aud_preds_from_logits(self, logits):
        return tf.nn.sigmoid(logits)


class EqOddsUnweightedGan(DemParGan):
    """Like DemParGan, but auditor gets to use the label Y as well"""
    def _get_aud_inputs(self):
            return tf.concat([self.Z, self.Y], axis=1)

    def _get_sensitive_logits(self, inputs, scope_name='model/aud', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            mlp = MLP(name='latents_to_sensitive_logits',
                      shapes=[self.zdim +  1 * self.ydim] + self.hidden_layer_specs['aud'] + [self.adim],
                      activ=ACTIV)
            return mlp.forward(inputs)


class EqoppUnweightedGan(DemParGan):
    """Like DemParGan, but only using Y = 0 examples"""
    def _get_loss(self):  # produce losses for the fairness task
        loss = self.class_coeff*self.class_loss + self.recon_coeff*self.recon_loss - self.fair_coeff*self.aud_loss
        eqopp_class_loss = tf.multiply(1.  - self.Y, loss)
        return tf.reduce_mean(eqopp_class_loss)


class WassGan(AbstractBaseNet):
    """Uses the Wasserstein loss function"""
    def _get_class_loss(self, Y_hat, Y):
        return wass_loss(Y, Y_hat)

    def _get_aud_loss(self, A_hat, A):
        return wass_loss(A, A_hat)

class WassGanNoSig(WassGan):
    """Uses the Wasserstein loss function"""
    def _get_class_preds_from_logits(self, logits):
        return logits

    def _get_aud_preds_from_logits(self, logits):
        return logits


class DemParWassGan(WassGan, DemParGan):
    """Like DemParGan, but wass_loss"""
    def _get_class_loss(self, Y_hat, Y):
        return WassGan._get_class_loss(self, Y_hat, Y)

    def _get_aud_loss(self, A_hat, A):
        return WassGan._get_aud_loss(self, A_hat, A)


class EqOddsUnweightedWassGan(WassGan, EqOddsUnweightedGan):
    """Like DemParGan, but auditor gets to use the label Y as well. And wass_loss"""
    def _get_class_loss(self, Y_hat, Y):
        return WassGan._get_class_loss(self, Y_hat, Y)

    def _get_aud_loss(self, A_hat, A):
        return WassGan._get_aud_loss(self, A_hat, A)


class WeightedGan(AbstractBaseNet):
    """
    Weighted means we are going to scale each component of the loss according to the
    relative size of the group (A=0 or A=1) it belongs to. this is how we implement a group-normalized loss fn
    """
    def __init__(self,
                 recon_coeff=RECON_COEFF,
                 class_coeff=CLASS_COEFF,
                 fair_coeff=FAIR_COEFF,
                 xdim=XDIM,
                 ydim=YDIM,
                 zdim=ZDIM,
                 adim=ADIM,
                 hidden_layer_specs=HIDDEN_LAYER_SPECS,
                 seed=SEED,
                 hinge=HINGE,
                 A_weights=A_WTS,
                 Y_weights=Y_WTS,
                 AY_weights=AY_WTS,
                 **kwargs):
        self.A_weights = A_weights
        self.Y_weights = Y_weights
        self.AY_weights = AY_weights

        super().__init__(recon_coeff, class_coeff, fair_coeff, xdim, ydim, zdim, adim, hidden_layer_specs, seed=seed, hinge=hinge, **kwargs)
        self.unweighted_aud_loss = self._get_aud_loss(self.A_hat, self.A)
        self.aud_loss = self._get_weighted_aud_loss(self.unweighted_aud_loss, self.A_weights, self.Y_weights, self.AY_weights)
        self.loss = self._get_loss()

    @abstractmethod
    def _get_weighted_class_loss(self, L, A_wts, Y_wts, AY_wts): #weight class loss for final loss function
        pass

    @abstractmethod
    def _get_weighted_recon_loss(self, L, A_wts, Y_wts, AY_wts): #weight recon loss for final loss function
        pass

    @abstractmethod
    def _get_weighted_aud_loss(self, L, A_wts, Y_wts, AY_wts): #weight auditor loss for final loss function
        pass


class WeightedDemParGan(WeightedGan, DemParGan):
    def _weight_loss(self, L, A_wts, Y_wts, AY_wts):
        A0_wt = A_wts[0]
        A1_wt = A_wts[1]
        wts = A0_wt * (1. - self.A) + A1_wt * self.A
        wtd_L = tf.multiply(L, tf.squeeze(wts))
        return wtd_L

    def _get_weighted_class_loss(self, L, A_wts, Y_wts, AY_wts):
        return self._weight_loss(L, A_wts, Y_wts, AY_wts)

    def _get_weighted_recon_loss(self, L, A_wts, Y_wts, AY_wts):
        return self._weight_loss(L, A_wts, Y_wts, AY_wts)

    def _get_weighted_aud_loss(self, L, A_wts, Y_wts, AY_wts):
        return self._weight_loss(L, A_wts, Y_wts, AY_wts)


class WeightedEqoddsGan(WeightedDemParGan, EqOddsUnweightedGan):
    def _weight_loss(self, L, A_wts, Y_wts, AY_wts):
        A0_Y0_wt = AY_wts[0][0]
        A0_Y1_wt = AY_wts[0][1]
        A1_Y0_wt = AY_wts[1][0]
        A1_Y1_wt = AY_wts[1][1]

        wts = A0_Y0_wt * tf.multiply(1 - self.A, 1 - self.Y) \
            + A0_Y1_wt * tf.multiply(1 - self.A, self.Y) \
            + A1_Y0_wt * tf.multiply(self.A, 1 - self.Y) \
            + A1_Y1_wt * tf.multiply(self.A, self.Y)
        wtd_L = tf.multiply(L, tf.squeeze(wts))

        return wtd_L

    def _get_sensitive_logits(self, latents, scope_name='model/aud', reuse=False):
        return EqOddsUnweightedGan._get_sensitive_logits(self, latents, scope_name, reuse)
        #return EqOddsUnweightedGan._get_sensitive_logits(self, latents, scope_name='model/aud', reuse=False)


class WeightedDemParWassGan(WassGan, WeightedDemParGan):
    def _get_class_loss(self, Y_hat, Y):
        return WassGan._get_class_loss(self, Y_hat, Y)

    def _get_aud_loss(self, A_hat, A):
        return WassGan._get_aud_loss(self, A_hat, A)


class WeightedEqoddsWassGan(WassGan, WeightedEqoddsGan):
    def _get_class_loss(self, Y_hat, Y):
        return WassGan._get_class_loss(self, Y_hat, Y)

    def _get_aud_loss(self, A_hat, A):
        return WassGan._get_aud_loss(self, A_hat, A)



class WeightedDemParWassGpGan(WeightedDemParWassGan):
    """
    gradient penalty style training;
    want the norm of the auditor gradients close to 1 in regions of Z space between the two groups,
    i.e. broaden the decision boundary to give useful gradients, and make the auditor 1-Lipshitz"""
    def __init__(self, *args, gp=10.0, batch_size=64, **kwargs):
        super(WeightedDemParWassGpGan, self).__init__(*args, **kwargs)
        self.gp = gp
        self.batch_size = batch_size
        self.grad_norms = self._gradient_penalty(self.X, self.A)
        self.aud_loss = self._get_weighted_aud_loss(self.unweighted_aud_loss, self.A_weights, self.Y_weights, self.AY_weights) + self.gp*self.grad_norms
        self.aud_err = classification_error(self.A, tf.cast(tf.greater(self.A_hat, 0), tf.float32))
        self.loss = self._get_loss()

    def _get_aud_preds_from_logits(self, logits):
        return logits

    def _gradient_penalty(self, X, A):
        def _xor(x, y):
            x_or_y = tf.minimum(x + y, 1.0)
            not_x_and_y = 1.0 - x*y
            return tf.squeeze(x_or_y*not_x_and_y)

        idx = tf.random_shuffle(tf.range(0, self.batch_size), seed=self.seed)
        Xs = tf.gather(X, idx)  # x shuffled
        As = tf.gather(A, idx)  # a shuffled
        u = tf.random_uniform(Xs.shape)
        #u = tf.random_uniform((self.batch_size, 1))
        Xmix = u*X + (1 - u)*Xs  # a minibatch of randomly convexly mixed X and Xs
        Ahat_mix = self._get_sensitive_logits(self._get_latents(Xmix, reuse=True), reuse=True)
        A_xor_As = _xor(self.A, As)  # one iff original and shuffled data belong to diff groups
        scale_factor = self.batch_size/tf.reduce_sum(A_xor_As)
        return scale_factor*tf.losses.mean_squared_error(
                labels=tf.ones([self.batch_size, ]),
                predictions=tf.norm(tf.gradients(Ahat_mix, Xmix)[0], axis=1),
                weights=A_xor_As  # we only care about getting grad norm == 1 where A_i != As_i 
                )

      
class WeightedDemParWassGpCeGan(WeightedDemParWassGpGan):
    """grad penalty with cross entropy classifier loss"""
    def _get_class_loss(self, Y_hat, Y):
        return cross_entropy(Y, Y_hat)

class WeightedEqoddsWassGpGan(WeightedDemParWassGpGan, WeightedEqoddsWassGan):
    def _gradient_penalty(self, X, A):
        def _xor(x, y):
            x_or_y = tf.minimum(x + y, 1.0)
            not_x_and_y = 1.0 - x*y
            return tf.squeeze(x_or_y*not_x_and_y)

        idx = tf.random_shuffle(tf.range(0, self.batch_size), seed=self.seed)
        Y = self.Y
        Xs = tf.gather(X, idx)  # x shuffled
        As = tf.gather(A, idx)  # a shuffled
        Ys = tf.gather(Y, idx)  # y shuffled
        u = tf.random_uniform((self.batch_size, 1))
        print(0, 'u', u.get_shape())
        Xmix = u*X + (1 - u)*Xs  # a minibatch of randomly convexly mixed X and Xs
        Ymix = u*Y + (1 - u)*Ys  # a minibatch of randomly convexly mixed Y and Ys
        Z_and_Y_mix = tf.concat((self._get_latents(Xmix, reuse=True), Ymix), axis=1)
        Ahat_mix = self._get_sensitive_logits(Z_and_Y_mix, reuse=True)
        A_xor_As = _xor(self.A, As)  # one iff original and shuffled data belong to diff groups
        scale_factor = self.batch_size/tf.reduce_sum(A_xor_As)
        return scale_factor*tf.losses.mean_squared_error(
                labels=tf.ones([self.batch_size, ]),
                predictions=tf.norm(tf.gradients(Ahat_mix, Z_and_Y_mix)[0], axis=1),
                weights=A_xor_As  # we only care about getting grad norm == 1 where A_i != As_i 
                )

    def _get_sensitive_logits(self, latents, scope_name='model/aud', reuse=False):
        return WeightedEqoddsWassGan._get_sensitive_logits(self, latents, scope_name, reuse)

    def _get_aud_preds_from_logits(self, logits):
        return WeightedDemParWassGpGan._get_aud_preds_from_logits(self, logits)
 
class WeightedEqoddsWassGpCeGan(WeightedEqoddsWassGpGan, WeightedDemParWassGpCeGan):
    def _get_class_loss(self, Y_hat, Y):
        return cross_entropy(Y, Y_hat)


# weighted eq opp are the same as their eq odds counter parts.
# you are required to set the Y = 1 weights to zero by hand, cf. src/laftr.py
WeightedEqoppGan = WeightedEqoddsGan
WeightedEqoppWassGan = WeightedEqoddsWassGan
WeightedEqoppWassGpGan = WeightedEqoddsWassGpGan
WeightedEqoppWassGpCeGan = WeightedEqoddsWassGpCeGan

     
class RegularizedFairClassifier(DemParGan):
    """
    disparate impact-regularized classifier
    we typically invoke this during transfer_learn.py with fair_coeff=0, i.e., naiver classifier
    """
    def _get_loss(self):
        def _get_fair_reg(Y, Y_hat, A):
            fpr0 = soft_rate(1 - Y, 1 - A, Y_hat)
            fpr1 = soft_rate(1 - Y, A, Y_hat)
            fnr0 = soft_rate(Y, 1 - A, Y_hat)
            fnr1 = soft_rate(Y, A, Y_hat)
            fpdi = tf.abs(fpr0 - fpr1)
            fndi = tf.abs(fnr0 - fnr1)

            di = 0.5 * (fpdi + fndi)
            return di

        return tf.reduce_mean([
            self.class_coeff * self.class_loss,
            0. * self.recon_loss,
            0. * self.aud_loss
        ]) + self.fair_coeff * _get_fair_reg(self.Y, self.Y_hat, self.A)


class RegularizedDPClassifier(RegularizedFairClassifier):
    """demographic parity regularized classifier"""
    def _get_loss(self):
        def _get_DP_reg(Y, Y_hat, A):
            pr0 = soft_rate_1(1 - A, Y_hat)
            pr1 = soft_rate_1(A, Y_hat)
            dp = tf.abs(pr0 - pr1)
            return dp

        return tf.reduce_mean([
            self.class_coeff * self.class_loss,
            0. * self.recon_loss,
            0. * self.aud_loss
        ]) + self.fair_coeff * _get_DP_reg(self.Y, self.Y_hat, self.A)

      

# model-specific utils
def cross_entropy(target, pred, weights=None, eps=EPS):
    if weights == None:
        weights = tf.ones_like(pred)
    return -tf.squeeze(tf.multiply(weights, tf.multiply(target, tf.log(pred + eps)) + tf.multiply(1 - target, tf.log(1 - pred + eps))))


def classification_error(target, pred):
    pred_class = tf.round(pred)
    return 1.0 - tf.reduce_mean(tf.cast(tf.equal(target, pred_class), tf.float32))


def wass_loss(target, pred):
    return tf.squeeze(tf.abs(target - pred))


def soft_rate(ind1, ind2, pred): #Y, A, Yhat
    mask = tf.multiply(ind1, ind2)
    rate = tf.reduce_sum(tf.multiply(tf.abs(pred - ind1), mask)) / tf.reduce_sum(mask + EPS)
    return rate


def soft_rate_1(ind1, pred): #Y, A, Yhat
    mask = ind1
    rate = tf.reduce_sum(tf.multiply(pred, mask)) / tf.reduce_sum(mask + EPS)
    return rate
