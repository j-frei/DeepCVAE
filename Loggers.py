from keras.backend import get_session, tf
from keras.callbacks import Callback


class Logger(Callback):
    def __init__(self, tb_logs, freq=2, log_file="./logs"):
        super(Callback, self).__init__()
        self.freq = freq
        self.log_file = log_file
        self.tb_logs = tb_logs
        self.sess = get_session()

        self.writer = tf.summary.FileWriter(log_file, self.sess.graph)

    def on_epoch_end(self, epoch, logs=None):
        # how many images do we want to store?
        use_batch_size = 5

        # validation_data = [input,output0,output1,...weights...]
        payload = {}
        payload['val_X'] = self.validation_data[0][:use_batch_size]
        payload['val_y'] = self.validation_data[1][:use_batch_size]
        payload['pred'] = self.model.predict(payload['val_X'])
        payload['batchsize'] = min(use_batch_size, len(payload['val_X']))
        for name,t,fn in self.tb_logs:
            d = fn(payload)
            if t == "text":
                tensor_txt = tf.convert_to_tensor(d,dtype=tf.string)
                summ_txt = tf.summary.text(name, tensor_txt)
                self.writer.add_summary(summ_txt.eval(session=self.sess), global_step=epoch)

            if t == "image":
                tensor_imgs = tf.convert_to_tensor(d)
                summ_imgs = tf.summary.image(name, tensor_imgs)
                self.writer.add_summary(summ_imgs.eval(session=self.sess), global_step=epoch)

        self.writer.flush()