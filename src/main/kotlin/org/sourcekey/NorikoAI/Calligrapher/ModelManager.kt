package org.sourcekey.NorikoAI.Calligrapher

import ExtendedFun.jsObject
import TensorFlowJS.LayersModel
import TensorFlowJS.SaveResult
import TensorFlowJS.Tensor
import TensorFlowJS.tf
import org.w3c.dom.History
import kotlin.js.Json
import kotlin.js.Promise


class ModelManager(private val project: Project, private val dataConverter: DataConverter) {

    private fun Array<Int>.selfMul(): Int {
        var number = 1
        this.forEach { number *= it }
        return number
    }

    data class ModelShape(
        var modelName: String,
        private val newModel: () -> LayersModel
    ) {
        var model: LayersModel? = null
        set(value) {
            // 加入最佳化的求解器、用MSE做為損失計算方式
            value?.compile(jsObject {
                optimizer = tf.train.adam()
                loss = "meanSquaredError"
                metrics = arrayOf("mse")
            })
            field = value
        }

        /**
         *
         * */
        fun load(): Promise<LayersModel> {
            val promiseModle = tf.loadLayersModel("indexeddb://${modelName}")
            promiseModle.then { model = it }.catch { model = newModel() }
            return promiseModle
        }

        /**
         *
         * */
        fun save(): Promise<SaveResult>? {
            return model?.save("indexeddb://${modelName}")
        }

        init {
            load()
        }
    }

    private val newModel1 = fun(): LayersModel {
        return tf.tidy {
            val model = tf.sequential()
            model.add(tf.layers.lstm(jsObject {
                units = 200
                returnSequences = false
                inputShape = dataConverter.inputShape
            }))
            model.add(tf.layers.dense(jsObject {
                units = dataConverter.outputUnits
                useBias = true
            }))
            model
        }
    }

    private val newModel2 = fun(): LayersModel {
        // Create a sequential model
        val model = tf.sequential()
        model.add(tf.layers.dropout(jsObject {
            rate = 0.2
            inputShape = dataConverter.inputShape
        })) //# dropout on the inputs
        // Add a single hidden layer
        model.add(tf.layers.dense(jsObject {
            units = 1
            useBias = true
        }))
        model.add(tf.layers.dense(jsObject {
            units = 30
            activation = "sigmoid"
        }))
        model.add(tf.layers.dense(jsObject {
            units = 30
            activation = "sigmoid"
        }))
        // Add an output layer
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
        }))
        return model
    }

    private val newModel3 = fun(): LayersModel {
        // Create a sequential model
        val model = tf.sequential()
        model.add(tf.layers.dropout(jsObject {
            rate = 0.2
            inputShape = arrayOf(dataShape.selfMul())
        })) //# dropout on the inputs
        //# this helps mimic noise or missing data
        model.add(tf.layers.dense(jsObject {
            units = 2048 //128
            inputDim = 2048 //784
            activation = "relu"
        }))
        model.add(tf.layers.dropout(jsObject {
            rate = 10 //0.5
        }))
        model.add(tf.layers.dense(jsObject {
            units = 128
            activation = "tanh"
        }))
        model.add(tf.layers.dropout(jsObject {
            rate = 10 //0.5
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataShape.last()
            activation = "sigmoid"
        }))

        return model
    }

    private val newModel4 = fun(): LayersModel {
        val model = tf.sequential()
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
            inputShape = dataConverter.inputShape
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits + 100
            useBias = true
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits + 100
            useBias = true
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
        }))
        return model
    }

    private val newModel5 = fun(): LayersModel {
        val model = tf.sequential()
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
            //inputLength = dataShape[dataShape.lastIndex - 2]
            //inputDim = dataShape[dataShape.lastIndex - 1]
            inputShape = dataConverter.inputShape
        }))
        for(i in 0..3){
            model.add(tf.layers.dense(jsObject {
                units = dataConverter.outputUnits + 500
                useBias = true
            }))
        }
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
        }))
        return model
    }

    private val newModel6 = fun(): LayersModel {
        val model = tf.sequential()
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
            //inputLength = dataShape[dataShape.lastIndex - 2]
            //inputDim = dataShape[dataShape.lastIndex - 1]
            inputShape = dataConverter.inputShape
        }))
        for(i in 0..11){
            model.add(tf.layers.permute(jsObject {
                dims = arrayOf(2, 1)
            }))
        }
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
        }))
        return model
    }

    private val newModel7 = fun(): LayersModel {
        val model = tf.sequential()
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
            inputShape = dataConverter.inputShape
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
        }))
        return model
    }

    private val newModel8 = fun(): LayersModel {
        val model = tf.sequential()
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            //inputLength = dataShape[dataShape.lastIndex - 2]
            //inputDim = dataShape[dataShape.lastIndex - 1]
            inputShape = dataConverter.inputShape
        }))
        model.add(tf.layers.simpleRNN(jsObject{
            units = 50
            useBias = true
            returnSequences = true
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
        }))
        return model
    }

    private val newModel9 = fun(): LayersModel {
        val model = tf.sequential()
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
            inputShape = dataConverter.inputShape
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits + 100
            useBias = true
        }))
        model.add(tf.layers.dense(jsObject {
            units = dataConverter.outputUnits
            useBias = true
        }))
        return model
    }

    /**
     *
     * */
    val models = run {
        val models = ArrayLinkList<ModelShape>()
        val modelName = project.name + "Model"
        models.add(ModelShape(modelName, newModel9))
        models
    }

    /**
     *
     * */
    val model: LayersModel?
        get() = models.node?.model

    /**
     *
     * */
    fun add(modelShape: ModelShape) {
        models.add(modelShape)
    }

    /**
     *
     * */
    fun remove(modelShape: ModelShape) {
        models.remove(modelShape)
    }

    /**
     *
     * */
    fun next() {
        models.next()
    }

    /**
     *
     * */
    fun previous() {
        models.previous()
    }

    /**
     *
     * */
    fun designated(nodeID: Int) {
        models.designated(nodeID)
    }

    /**
     *
     * */
    fun rename(modelName: String) {
        models.node?.modelName = modelName
    }

    /**
     *
     * */
    fun save(): Promise<SaveResult>? {
        return models.node?.save()
    }

    /**
     *
     * */
    private fun Array<String>.renameRepeatName(): Array<String> {
        val script = js(
            """
            function (arr){
              var count = {};
              arr.forEach(function(x,i) {
                if ( arr.indexOf(x) !== i ) {
                  var c = x in count ? count[x] = count[x] + 1 : count[x] = 1;
                  var j = c + 1;
                  var k = x + '(' + j + ')';
                  while( arr.indexOf(k) !== -1 ) k = x + '(' + (++j) + ')';
                  arr[i] = k;
                }
              });
              return arr;
            }
        """
        ) as (Array<String>) -> Array<String>
        return script(this)
    }

    /**
     *
     * */
    private fun String.toAvailableName(): String {
        val names = models.map(fun(it: ModelShape): String { return it.modelName }).toMutableList()
        names.add(this)
        return names.toTypedArray().renameRepeatName().getOrNull(names.lastIndex) ?: ""
    }

    /**
     *
     * */
    fun saveAs(modelName: String? = null): Promise<SaveResult>? {
        val model = models.node?.copy() ?: return null
        model.modelName = modelName ?: model.modelName.toAvailableName()
        models.add(model)
        models.designated(model)
        return model.save()
    }

    /**
     * 訓練模形
     * */
    fun train(inputs: Tensor, labels: Tensor, args: Json): Promise<History>? {
        return tf.tidy {
            model?.fit(inputs, labels, args)
        }
    }

    /**
     * 預測模形
     * */
    fun predict(inputs: Tensor): Tensor? {
        return tf.tidy {
            model?.predict(inputs)
        }
    }

}