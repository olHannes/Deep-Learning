
import tensorflow as tf

class Distiller(tf.keras.Model):
    def __init__(
            self,
            student,
            teacher,
            alpha=0.2,
            temperature=4.0,
            from_logits=False
    ):
        super().__init__()

        self.student = student
        self.teacher = teacher

        #Teacher should not be trained
        self.teacher.trainable = False

        self.alpha          = alpha
        self.temperature    = temperature
        self.from_logits    = from_logits

        self.total_loss_tracker         = tf.keras.metrics.Mean(name="loss")
        self.student_loss_tracker       = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker  = tf.keras.metrics.Mean(name="distillation_loss")

        self.student_metrics = []


    def compile(
            self,
            optimizer,
            student_loss_function,
            distillation_loss_function,
            metrics=None,
            **kwargs
        ):
        super().compile(optimizer=optimizer, **kwargs)

        self.student_loss_function      = student_loss_function
        self.distillation_loss_function = distillation_loss_function
        self.student_metrics            = metrics or []


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            *self.student_metrics,
        ]


    def call(self, inputs, training=False):
        return self.student(inputs, training=training)
    

    def _soft_targets(self, predictions):
        logits = None
        if self.from_logits:
            logits = predictions
        else:
            predictions = tf.clip_by_value(predictions, 1e-7, 1.0)
            logits = tf.math.log(predictions)
        return tf.nn.softmax(logits / self.temperature, axis=-1)
    

    def _compute_losses(self, x, y, sample_weight=None, training=False):
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=training)

        student_loss = self.student_loss_function(y, student_predictions, sample_weight=sample_weight)

        teacher_soft = self._soft_targets(teacher_predictions)
        student_soft = self._soft_targets(student_predictions)

        distillation_loss = self.distillation_loss_function(teacher_soft, student_soft, sample_weight=sample_weight)
        distillation_loss *= self.temperature **2

        total_loss = (self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss)

        return total_loss, student_loss, distillation_loss, student_predictions


    
    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            total_loss, student_loss, distillation_loss, student_predictions = (
                self._compute_losses(x, y, sample_weight=sample_weight, training=True)
            )
        
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)

        for metric in self.student_metrics:
            metric.update_state(y, student_predictions, sample_weight=sample_weight)

        return {metric.name: metric.result() for metric in self.metrics}
    
    
    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        total_loss, student_loss, distillation_loss, student_predictions = (
            self._compute_losses(x, y, sample_weight=sample_weight, training=False)
        )

        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)

        for metric in self.student_metrics:
            metric.update_state(y, student_predictions, sample_weight=sample_weight)

        return {metric.name: metric.result() for metric in self.metrics}