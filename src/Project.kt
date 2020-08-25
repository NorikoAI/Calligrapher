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
@JsModule("src/font/SourceHanSans_v1.001/SourceHanSansTC-Regular.otf") external val sourceHanSansTCUrl: String


data class Project(
        val projectName: String,
        val produceFontUrl: String,
        val referenceFontUrl: String = sourceHanSansTCUrl
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

    private fun Font.getGlyphByUnicode(unicode: Int): Glyph?{
        return this.glyphs.glyphs.toArray().find { glyph -> glyph.unicode == unicode }
    }

    /**
     *
     * */
    data class GlyphContrast(val referenceGlyph: Glyph, val produceGlyph: Glyph)

    /**
     *
     * */
    fun Array<GlyphContrast>.getOrNullByUnicode(unicode: Int): GlyphContrast?{
        return this.find{ glyphContrast -> glyphContrast.referenceGlyph.unicode == unicode }
    }

    /**
     *
     * */
    data class TrainingSet(val x: Array<Glyph>, val y: Array<Glyph>)

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
            char.referenceGlyph != null && char.referenceGlyph.path.toPathData() !== "" && char.referenceGlyph.unicode != null &&
                    char.produceGlyph != null && char.produceGlyph.path.toPathData() !== "" && char.produceGlyph.unicode != null
        }.toTypedArray()
    }

    fun xx(){
        println(getMatchedCharList(referenceFont, produceFont)[130].referenceGlyph.unicode)
    }

    /**
     * 記錄字形輪廓線嘅字串長度
     *
     * 因TensorflowJS需要每次輸入嘅數據形狀必要一樣
     * 所以要統一所有字串長度
     * **此值必須大於最長字形輪廓線嘅字串長度**
     */
    private val glyphPathStringLength = 16384

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
    data class TrainingTensorSet(
            val inputs: Tensor,
            val labels: Tensor,
            val inputMax: Tensor,
            val inputMin: Tensor,
            val labelMax: Tensor,
            val labelMin: Tensor
    )

    /**
     *
     * */
    private fun getTrainData(referenceFont: Font, produceFont: Font, numberOfDataForTrain: Int? = null):Array<TrainingSet> {
        //匯入 要參照字形嘅檔 同 要產生字體嘅檔
        val trainingData = getMatchedCharList(referenceFont, produceFont)
        //
        val drawedNumbers = Random.drawNumbersOfRange(0, trainingData.size, numberOfDataForTrain?:trainingData.size)
        println(drawedNumbers.toString())
        //
        var temporaryTrainingDate = ArrayList<TrainingSet>()
        drawedNumbers.forEach{index ->
            console.log("index ${index}")
            val referenceChar = trainingData.getOrNull(index)?:return@forEach
            //收集每一定數量就return出去訓練住先
            val datalength = 100// trainingData.length
            trainingData.forEachIndexed{i, produceChar ->
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
        /*drawedNumbers.forEach{index ->
            console.log("index ${index}")
            val referenceChar = trainingData.getOrNull(index)?:return@forEach
            //收集每一定數量就return出去訓練住先
            val datalength = 100// trainingData.length
            var temporaryTrainingDate = ArrayList<TrainingSet>()
            trainingData.forEachIndexed{i, produceChar ->
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
                    println("${(i%numberOfTrainingSessions) == (numberOfTrainingSessions-1)} ${i == datalength-1}")
                    console.log("return"+temporaryTrainingDate.size)
                    //return出去訓練住先
                    onGet(temporaryTrainingDate.toTypedArray())
                    //清空
                    temporaryTrainingDate = ArrayList()
                }
            }
            console.log(index.toString()+"done")
        }*/
    }

    /**
     *
     * */
    private fun String.uniformLength(): Array<Int>{
        val charArray = ArrayList<Int>()
        var i = 0
        while(i < glyphPathStringLength ){
            if(this.getOrNull(i) != null){
                charArray.add(this.codePointAt(i)?:0)
            }else{
                charArray.add(0)
            }
            i++
        }
        return charArray.toTypedArray()
    }

    /**
     *
     * */
    private fun Array<Int>.ununiformLength(): String{
        var string = ""
        for(i in 0 until this.size){
            string += String.fromCharPoint(this[i])
        }
        return string
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
                       d.x[0].path.toPathData().uniformLength(),
                       d.x[1].path.toPathData().uniformLength(),
                       d.x[2].path.toPathData().uniformLength()
                )
            }.toTypedArray()
            val labels = data.map{d ->
                arrayOf(
                       d.y[0].path.toPathData().uniformLength(),
                       d.y[1].path.toPathData().uniformLength(),
                       d.y[2].path.toPathData().uniformLength()
                )
            }.toTypedArray()
            val inputTensor = tf.tensor3d(inputs, arrayOf(inputs.size, 3, glyphPathStringLength))!!
            val labelTensor = tf.tensor3d(labels, arrayOf(labels.size, 3, glyphPathStringLength))!!
            //取最大值與最小值
            val inputMax = inputTensor.max()
            val inputMin = inputTensor.min()
            val labelMax = labelTensor.max()
            val labelMin = labelTensor.min()
            //正規化 將 (tensor內的資料-最小值)/(最大值-最小值)) 出來的結果在0-1之間
            val normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
            val normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

            return TrainingTensorSet(
                    normalizedInputs,
                    normalizedLabels,
                    inputMax,
                    inputMin,
                    labelMax,
                    labelMin
            )
        })
    }

    /**
     * 建立模型
     * */
    private fun newModel(): LayersModel {
        // Create a sequential model
        val model = tf.sequential()
        // Add a single hidden layer
        model.add(tf.layers.dense(jsObject{
            inputShape = arrayOf(3, glyphPathStringLength)
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
            units = glyphPathStringLength
            useBias = true
        }))
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
                model = tf.loadLayersModel("indexeddb://project-model-${projectName}").await()
            }
        }
        newModel()
    }()

    /**
     * 每次訓練的樣本數
     * */
    private val batchSize = 32

    /**
     * 訓練多少代
     * */
    private val epochs = 30

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
        val inputMax = normalizationData.inputMax
        val inputMin = normalizationData.inputMin
        val labelMin = normalizationData.labelMin
        val labelMax = normalizationData.labelMax
        return tf.tidy(fun(): Array<Array<dynamic>>{
            //tf.linspace(start_value,end_value,number_of_value)
            val input_x = tf.linspace(0, 1, 100)
            //將產生的資料轉成[num_examples, num_features_per_example]
            val preds = model.predict(normalizationData.inputs)
            //轉回原本的數= 數字*(最大值-最小值)+最小值
            val toOrignalX = input_x.mul(inputMax.sub(inputMin)).add(inputMin)
            val toOrignalY = preds.mul(labelMax.sub(labelMin)).add(labelMin)
            //tensor.dataSync() return data from tensor to array
            return arrayOf(toOrignalX.dataSync(), toOrignalY.dataSync())
        })
    }

    fun trainModel(){
        GlobalScope.launch {
            val data = getTrainData(referenceFont, produceFont,1)
            val tensorData = convertToTensor(data)
            trainModel(model, tensorData.inputs, tensorData.labels).await()
            println(JSON.stringify(getPrediction(tensorData)))
        }
    }

    fun saveModel(){
        model.save("indexeddb://project-model-${projectName}")
    }

    fun loadModel(){
        GlobalScope.launch {
            model = tf.loadLayersModel("indexeddb://project-model-${projectName}").await()
        }
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
        val input = arrayOf(
                produceChar.toPathData().uniformLength(),
                referenceChar.toPathData().uniformLength(),
                referenceGlyph.toPathData().uniformLength()
        )
        val output = this.predict(tf.tensor(input, arrayOf(3, glyphPathStringLength))!!).dataSync()
        //val orignal = output.mul(inputMax.sub(inputMin)).add(inputMin)
        val produceGlyphSvgPath = (Object.values(output)[0] as Array<Int>).ununiformLength()
        return Path().setCommands(produceGlyphSvgPath)
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
        val produceChar = referenceFont.getGlyphByUnicode(produceGlyphUnicode)?:return null
        val referenceChar = referenceFont.getGlyphByUnicode(referenceGlyphUnicode)?:return null
        val referenceGlyph = produceFont.getGlyphByUnicode(referenceGlyphUnicode)?:return null
        return model.produceGlyph(produceChar, referenceChar, referenceGlyph)
    }





    init {

    }
}