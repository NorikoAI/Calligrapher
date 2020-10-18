package org.sourcekey.NorikoAI.Calligrapher

import app.*
import kotlinext.js.Object
import kotlinx.browser.window
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import org.w3c.dom.History
import kotlin.js.Promise
import kotlin.js.json
import kotlin.random.Random
import org.sourcekey.NorikoAI.Calligrapher.OpentypeJS.Font
import org.sourcekey.NorikoAI.Calligrapher.OpentypeJS.Glyph
import org.sourcekey.NorikoAI.Calligrapher.OpentypeJS.Path
import react.RBuilder
import react.dom.button
import react.dom.div
import react.dom.svg

@JsModule("src/font/SourceHanSans_v1.001/SourceHanSansTC-Regular.otf") external val sourceHanSansTCUrl: String


data class Project(
        val projectName: String,
        private val produceFontUrl: String,
        private val referenceFontUrl: String = sourceHanSansTCUrl
) {

    /**
     *
     * */
    var referenceFont: Font = {
        GlobalScope.launch { referenceFont = OpentypeJS.load(referenceFontUrl).await() }
        Font(jsObject{
            familyName = "null"
            styleName = "null"
            unitsPerEm = 1000
            ascender = 800
            descender = -200
        })
    }()
    private set

    /**
     *
     * */
    var produceFont: Font = {
        GlobalScope.launch { produceFont = OpentypeJS.load(produceFontUrl).await() }
        Font(jsObject{
            familyName = "null"
            styleName = "null"
            unitsPerEm = 1000
            ascender = 800
            descender = -200
        })
    }()
    private set

    private fun Font.getOrNullGlyphByUnicode(unicode: Int): Glyph?{
        return this.glyphs.glyphs.toArray().find { glyph -> glyph.unicode == unicode }
    }

    private val Glyph.reference: Glyph?
        get() = referenceFont.glyphs.glyphs.toArray().find{ g -> g.unicode == this.unicode }

    /**
     *
     * */
    private data class GlyphContrast(val referenceGlyph: Glyph, val produceGlyph: Glyph)

    /**
     *
     * */
    private data class TrainingSet(val x: Array<Glyph>, val y: Array<Glyph>)

    /**
     *
     * */
    private fun getMatchedCharList(referenceFont: Font, produceFont: Font): Array<GlyphContrast>{
        //匯入 要參照字形嘅檔 同 要產生字體嘅檔
        val referenceGlyphs = referenceFont.glyphs.glyphs.toArray()
        val produceGlyphs = produceFont.glyphs.glyphs.toArray()
        //以Unicode對稱所有字 同 篩走缺少嘅字
        return referenceGlyphs.map{charR ->
            GlyphContrast(
                    charR,
                    produceGlyphs.find{charP -> charP.unicode === charR.unicode}?: Glyph(js("{}"))
            )
        }.filter{char ->
            char.referenceGlyph.path.commands.isNotEmpty() && char.referenceGlyph.unicode != null &&
                    char.produceGlyph.path.commands.isNotEmpty() && char.produceGlyph.unicode != null
        }.toTypedArray()
    }

    /**
     * 記錄字形輪廓線嘅字串長度
     *
     * 因TensorflowJS需要每次輸入嘅數據形狀必要一樣
     * 所以要統一所有字串長度
     * **此值必須大於最長字形輪廓線嘅字串長度**
     */
    private val glyphPathCommandLength = 256

    /**
     * 每次訓練數量
     *
     * 為左避免匯入大量數據作訓練而造成超載
     * 所以就續少訓練再匯入訓練數據再訓練
     */
    private val numberOfTrainingSessions = 1000

    /**
     *
     * */
    data class TrainingTensorSet(val inputs: Tensor, val labels: Tensor)

    private val dataShape: Array<Int> = arrayOf(3, glyphPathCommandLength, 12)

    private val dataConverter = DataConverter(dataShape)

    /**
     *
     * */
    private fun Array<GlyphContrast>.getTrainData(indexs: Array<Int>, onGet: (Array<TrainingSet>)-> Unit){
        var temporaryTrainingDate = ArrayList<TrainingSet>()
        indexs.forEach{index ->
            val referenceChar = this.getOrNull(index)?:return@forEach
            //收集每一定數量就return出去訓練住先
            val datalength = 200// trainingData.length
            this.forEachIndexed{i, produceChar ->
                temporaryTrainingDate.add(TrainingSet(
                        arrayOf(
                                produceChar.referenceGlyph,
                                referenceChar.referenceGlyph,
                                referenceChar.produceGlyph
                        ),
                        arrayOf(
                                produceChar.produceGlyph,
                                referenceChar.referenceGlyph,
                                referenceChar.produceGlyph
                        )
                ))
                if((i%numberOfTrainingSessions) == (numberOfTrainingSessions-1) || i >= datalength-1){
                    console.log("return"+temporaryTrainingDate.size)
                    //return出去訓練住先
                    onGet(temporaryTrainingDate.toTypedArray())
                    //清空
                    temporaryTrainingDate = ArrayList()
                }
            }
            console.log(index.toString()+"done")
        }
    }

    /**
     *
     * */
    private fun Array<GlyphContrast>.getTrainData(indexs: Array<Int>): Array<TrainingSet> {
        var temporaryTrainingDate = ArrayList<TrainingSet>()
        indexs.forEach{index ->
            val referenceChar = this.getOrNull(index)?:return@forEach
            //收集每一定數量就return出去訓練住先
            val datalength = 100// trainingData.length
            this.forEachIndexed{i, produceChar ->
                temporaryTrainingDate.add(TrainingSet(
                        arrayOf(
                                produceChar.referenceGlyph,
                                referenceChar.referenceGlyph,
                                referenceChar.produceGlyph
                        ),
                        arrayOf(
                                produceChar.produceGlyph,
                                referenceChar.referenceGlyph,
                                referenceChar.produceGlyph
                        )
                ))
                if((i%numberOfTrainingSessions) == (numberOfTrainingSessions-1) || i >= datalength-1){
                    console.log("return"+temporaryTrainingDate.size)
                    //return出去訓練住先
                    return temporaryTrainingDate.toTypedArray()
                    //清空
                    temporaryTrainingDate = ArrayList()
                }
            }
            console.log(index.toString()+"done")
        }
        return temporaryTrainingDate.toTypedArray()
    }

    private fun Array<GlyphContrast>.getTrainData(numberOfDataForTrain: Int? = null): Array<TrainingSet> {
        return getTrainData(Random.drawNumbersOfRange(
                0, this.size,
                numberOfDataForTrain?:this.size
        ))
    }

    private fun Array<GlyphContrast>.getTrainDataByUnicode(unicodes: Array<Int>): Array<TrainingSet> {
        val indexs = ArrayList<Int>()
        this.forEachIndexed { index, glyphContrast ->
            if(glyphContrast.referenceGlyph.unicode?.equals(unicodes) == true){
                indexs.add(index)
            }
        }
        return getTrainData(indexs.toTypedArray())
    }

    private fun Array<GlyphContrast>.getTrainDataByUnicode(vararg unicodes: Int): Array<TrainingSet> {
        return getTrainDataByUnicode(unicodes.toTypedArray())
    }

    /**
     * 把資料轉成Tensor
     * */
    private fun convertToTensor(data: Array<TrainingSet>): TrainingTensorSet {
        // 使用tf.tidy讓除了回傳值以外，中間過程中的所佔用的空間釋放掉
        return tf.tidy(fun(): TrainingTensorSet  {
            // 打亂資料，在訓練最好都要做打亂資料的動作
            tf.util.shuffle(data)
            // 將資料轉成tensor
            val inputs = data.map{d ->
                arrayOf(
                       d.x[0].path.commands,
                       d.x[1].path.commands,
                       d.x[2].path.commands
                )
            }.toTypedArray()
            val labels = data.map{d ->
                arrayOf(
                       d.y[0].path.commands,
                       d.y[1].path.commands,
                       d.y[2].path.commands
                )
            }.toTypedArray()
            val inputTensor = dataConverter.encodeTensor(inputs)
            val labelTensor = dataConverter.encodeTensor(labels)

            return TrainingTensorSet(inputTensor, labelTensor)
        })
    }

    /**
     * 建立模型
     * */
    private fun newModel(): LayersModel {
        // Create a sequential model
        val model = tf.sequential()
        model.add(tf.layers.dropout(jsObject {
            rate = 0.2
            inputShape = dataShape
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
        /*
        // Add a single hidden layer
        model.add(tf.layers.dense(jsObject{
            inputShape = dataShape
            units = 1
            useBias = true
        }))
        model.add(tf.layers.dense(jsObject{
            units = 30
            activation = "sigmoid"
        }))
        model.add(tf.layers.dense(jsObject{
            units = 30
            activation = "sigmoid"
        }))
        // Add an output layer
        model.add(tf.layers.dense(jsObject{
            units = dataShape.last()
            useBias = true
        }))*/
        // 加入最佳化的求解器、用MSE做為損失計算方式
        model.compile(jsObject{
            optimizer = tf.train.adam()
            loss = "meanSquaredError"
            metrics = arrayOf("mse")
        })

        return model
    }

    private var isInitModel: Boolean = false

    /**
     *
     * */
    private var model: LayersModel = {
        if(isInitModel){
            GlobalScope.launch {
                //model = tf.loadLayersModel("indexeddb://project-model-${projectName}").await()
            }
        }
        newModel()
    }()

    /**
     * 每次訓練的樣本數
     * */
    private val batchSize = 320

    /**
     * 訓練多少代
     * */
    private val epochs = 50

    /**
     * 訓練的程式碼
     * */
    private fun trainModel(model: LayersModel, inputs: Tensor, labels: Tensor): Promise<History> {
        return model.fit(inputs, labels, jsObject {
            this.batchSize = batchSize
            this.epochs = epochs
            shuffle = true
            callbacks = tfvis.show.fitCallbacks(
                    jsObject{ name = "Training Performance"},
                    arrayOf("loss", "mse"),
                    jsObject{
                        height = 200
                        callbacks = arrayOf("onEpochEnd")
                    }
            )
        })
    }

    fun getPrediction(normalizationData: TrainingTensorSet): Array<Array<dynamic>> {
        val inputMax = normalizationData.inputs.max()
        val inputMin = normalizationData.inputs.min()
        val labelMax = normalizationData.labels.max()
        val labelMin = normalizationData.labels.min()
        return tf.tidy(fun(): Array<Array<dynamic>>{
            //tf.linspace(start_value,end_value,number_of_value)
            val input_x = tf.linspace(0, 1, 100)
            //將產生的資料轉成[num_examples, num_features_per_example]
            val preds = model.predict(normalizationData.inputs)
            /*
            //轉回原本的數= 數字*(最大值-最小值)+最小值
            val toOrignalX = input_x.mul(inputMax.sub(inputMin)).add(inputMin)
            val toOrignalY = preds.mul(labelMax.sub(labelMin)).add(labelMin)
            //tensor.dataSync() return data from tensor to array
            return arrayOf(toOrignalX.dataSync(), toOrignalY.dataSync())
            */
            //轉回原本的數= 數字*(最大值-最小值)+最小值
            val toOrignalX = dataConverter.decodeTensor(input_x)
            val toOrignalY = dataConverter.decodeTensor(preds)
            //tensor.dataSync() return data from tensor to array
            return arrayOf(toOrignalX, toOrignalY)
        })
    }

    fun initModel(){
        model = newModel()
    }

    suspend fun trainModel(){
        val data = getMatchedCharList(referenceFont, produceFont).getTrainData(arrayOf(268))
        val tensorData = convertToTensor(data)
        trainModel(model, tensorData.inputs, tensorData.labels).await()
    }

    suspend fun saveModel(){
        model.save("indexeddb://project-model-${projectName}").await()
    }

    suspend fun loadModel(){
        model = tf.loadLayersModel("indexeddb://project-model-${projectName}").await()
    }

    /**
     * 產生要嘅字形字符
     *
     * 依
     *
     * @return produceGlyph
     *
     * ==========================================
     * |      | 字形1           | 字形2
     * |-----------------------------------------
     * | 字符1 | produceChar    | produceGlyph
     * | 字符2 | referenceChar  | referenceGlyph
     * ==========================================
     *
     * ==========================================
     * | 輸入             | 輸出
     * |-----------------------------------------
     * | [                | [
     * | produceChar,     | produceGlyph,
     * | referenceChar,   | referenceChar,(無用)
     * | referenceGlyph,  | referenceGlyph,(無用)
     * | ]                | ]
     * ==========================================
     *
     */
    private fun LayersModel.produceGlyph(produceChar: Path, referenceChar: Path, referenceGlyph: Path): Path{
        val input = arrayOf(arrayOf(
                    produceChar.commands,
                    referenceChar.commands,
                    referenceGlyph.commands
        ))
        val inputTensor = dataConverter.encodeTensor(input)
        val outputTensor = this.predict(inputTensor)
        val output = dataConverter.decodeTensor(outputTensor)
        val produceGlyphSvgPath = output[0][0]
        val path = Path()
        path.commands = produceGlyphSvgPath
        return path
    }

    /**
     *
     * */
    private fun LayersModel.produceGlyph(produceChar: Glyph, referenceChar: Glyph, referenceGlyph: Glyph): Glyph{
        return Glyph(jsObject{
            unicode = produceChar.unicode
            path = this@produceGlyph.produceGlyph(produceChar.path, referenceChar.path, referenceGlyph.path)
        })
    }

    /**
     *
     * */
    fun produceGlyph(produceGlyphUnicode: Int, referenceGlyphUnicode: Int): Glyph?{
        val produceChar = referenceFont.getOrNullGlyphByUnicode(produceGlyphUnicode)?:return null
        println(produceChar.path.toPathData())
        val referenceChar = referenceFont.getOrNullGlyphByUnicode(referenceGlyphUnicode)?:return null
        println(referenceChar.path.toPathData())
        val referenceGlyph = produceFont.getOrNullGlyphByUnicode(referenceGlyphUnicode)?:return null
        println(referenceGlyph.path.toPathData())
        return model.produceGlyph(produceChar, referenceChar, referenceGlyph)
    }





    init {

    }
}