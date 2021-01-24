package org.sourcekey.NorikoAI.Calligrapher

import ExtendedFun.jsObject
import ExtendedFun.range
import OpentypeJS.Font
import OpentypeJS.Glyph
import OpentypeJS.Path
import TensorFlowJS.Tensor
import TensorFlowJS.TensorflowVisualization
import TensorFlowJS.tfvis
import kotlinx.browser.document
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import org.w3c.dom.*
import kotlin.js.Json
import kotlin.random.Random


class Calligrapher(
    private val project: Project,
    private val getReferenceFont: ()-> Font,
    private val getFont: ()-> Font
) {
    /**
     *
     * */
    val font: Font
        get() = getFont()

    /**
     *
     * */
    private val referenceFont: Font
        get() = getReferenceFont()

    /**
     *
     * */
    var Glyph.referenceGlyph: Glyph?
        get() = asDynamic().referenceGlyph as? Glyph?: run{
            if(unicode != null){
                asDynamic().referenceGlyph = referenceFont.getGlyphByUnicode(
                    unicode?.toInt()?: return@run null
                )
            }
            asDynamic().referenceGlyph as? Glyph
        }
        set(value) { asDynamic().referenceGlyph = value }

    /**
     *
     * */
    private val dataConverter = DataConverter()

    /**
     * 每次訓練數量
     *
     * 為左避免匯入大量數據作訓練而造成超載
     * 所以就續少訓練再匯入訓練數據再訓練
     */
    var sectionSize = 30

    /**
     *
     * */
    private fun getSequenceTrainingData(
        sectionSize: Int = this.sectionSize,
        onGet: suspend (inputs: Array<Array<Array<Json>>>, labels: Array<Array<Array<Json>>>)->Unit
    ){
        GlobalScope.launch {
            val filterNotForLearningFont = ArrayList<Glyph>()
            for(glyph in font){ if(glyph?.filterNotKeep()?.filterInvalid() != null){
                filterNotForLearningFont.add(glyph)
            } }
            var inputsSection = ArrayList<Array<Array<Json>>>()
            var labelsSection = ArrayList<Array<Array<Json>>>()
            for (char2 in filterNotForLearningFont) {
                for ((i, char1) in filterNotForLearningFont.withIndex()) {
                    //收集部分訓練資料
                    /**
                     * ==========================================
                     * |      | 字形1           | 字形2
                     * |-----------------------------------------
                     * | 字符1 | char1glyph1    | char1glyph2
                     * | 字符2 | char2glyph1    | char2glyph2
                     * ==========================================
                     *
                     * ==========================================
                     * | 輸入             | 輸出
                     * |-----------------------------------------
                     * | [                | [
                     * | char1glyph1,     | char1glyph2,
                     * | char2glyph1,     | char2glyph1,(無用)
                     * | char2glyph2,     | char2glyph2,(無用)
                     * | ]                | ]
                     * ==========================================
                     */
                    val char1glyph1 = char1.referenceGlyph?: break
                    val char1glyph2 = char1
                    val char2glyph1 = char2.referenceGlyph?: break
                    val char2glyph2 = char2
                    inputsSection.add(arrayOf(
                        char1glyph1.path.commands,
                        char2glyph1.path.commands,
                        char2glyph2.path.commands
                    ))
                    labelsSection.add(arrayOf(
                        char1glyph2.path.commands,
                        char2glyph1.path.commands,
                        char2glyph2.path.commands
                    ))
                    //return部分訓練資料
                    if (i % sectionSize >= sectionSize - 1 || i >= filterNotForLearningFont.lastIndex) {
                        // 打亂資料，在訓練最好都要做打亂資料的動作
                        //tf.util.shuffle(inputsSection)
                        //tf.util.shuffle(labelsSection)
                        //return
                        onGet(inputsSection.toTypedArray(), labelsSection.toTypedArray())
                        //清空
                        inputsSection = ArrayList()
                        labelsSection = ArrayList()
                    }
                }
            }
        }
    }

    /**
     *
     * */
    private fun getTrainingData(
        sectionSize: Int = this.sectionSize,
        onGet: suspend (inputs: Array<Array<Array<Json>>>, labels: Array<Array<Array<Json>>>)->Unit
    ){
        GlobalScope.launch {
            val filterNotForLearningFont = ArrayList<Glyph>()
            for(glyph in font){ if(glyph?.filterNotKeep()?.filterInvalid() != null){
                filterNotForLearningFont.add(glyph)
            } }
            var inputsSection = ArrayList<Array<Array<Json>>>()
            var labelsSection = ArrayList<Array<Array<Json>>>()
            for ((i, char1) in filterNotForLearningFont.withIndex()) {
                //收集部分訓練資料
                /**
                 * ==========================================
                 * |      | 字形1           | 字形2
                 * |-----------------------------------------
                 * | 字符1 | char1glyph1    | char1glyph2
                 * | 字符2 | char2glyph1    | char2glyph2
                 * ==========================================
                 *
                 * ==========================================
                 * | 輸入             | 輸出
                 * |-----------------------------------------
                 * | [                | [
                 * | char1glyph1,     | char1glyph2,
                 * | char2glyph1,     | char2glyph1,(無用)
                 * | char2glyph2,     | char2glyph2,(無用)
                 * | ]                | ]
                 * ==========================================
                 */
                val randomNumber = Random.range(0, filterNotForLearningFont.size)?: break
                val char2 = filterNotForLearningFont.getOrNull(randomNumber)?: break
                val char1glyph1 = char1.referenceGlyph?: break
                val char1glyph2 = char1
                val char2glyph1 = char2.referenceGlyph?: break
                val char2glyph2 = char2
                inputsSection.add(arrayOf(
                        char1glyph1.path.commands,
                        char2glyph1.path.commands,
                        char2glyph2.path.commands
                ))
                labelsSection.add(arrayOf(
                        char1glyph2.path.commands,
                        char2glyph1.path.commands,
                        char2glyph2.path.commands
                ))
                //return部分訓練資料
                if (i % sectionSize >= sectionSize - 1 || i >= filterNotForLearningFont.lastIndex) {
                    // 打亂資料，在訓練最好都要做打亂資料的動作
                    //tf.util.shuffle(inputsSection)
                    //tf.util.shuffle(labelsSection)
                    //return
                    onGet(inputsSection.toTypedArray(), labelsSection.toTypedArray())
                    //清空
                    inputsSection = ArrayList()
                    labelsSection = ArrayList()
                }
            }
        }
    }

    /**
     *
     * */
    private fun getPredictingData(
        predictUnicodes: Array<Int>,
        onGet: (input: Array<Array<Array<Json>>>, unicode: Int)->Unit
    ){
        for(char1Unicode in predictUnicodes){
            val filterNotForLearningFont = ArrayList<Glyph>()
            for(glyph in font){ if(glyph?.filterNotKeep()?.filterInvalid() != null){
                filterNotForLearningFont.add(glyph)
            } }
            val char1 = font.getGlyphByUnicode(char1Unicode)?: break
            val randomNumber = Random.range(0, filterNotForLearningFont.size)?: break
            val char2 = filterNotForLearningFont.getOrNull(randomNumber)?: break
            val char1glyph1 = char1.referenceGlyph?: break
            val char2glyph1 = char2.referenceGlyph?: break
            val char2glyph2 = char2
            onGet(arrayOf(arrayOf(
                char1glyph1.path.commands,
                char2glyph1.path.commands,
                char2glyph2.path.commands
            )), char1Unicode)
        }
    }

    /**
     *
     * */
    val modelManager = ModelManager(project, dataConverter)

    /*
    private suspend fun renderImage(container: HTMLElement, tensor: Tensor, imageOpts: dynamic) {
        var resized = tf.tidy{
            tf.image.resizeNearestNeighbor(
                tensor,
                arrayOf(imageOpts.height as Int, imageOpts.width as Int)
            ).clipByValue(0.0, 1.0)
        }
        var canvas = (container.querySelector("canvas")?: document.createElement("canvas")) as HTMLCanvasElement
        canvas.width = imageOpts.width
        canvas.height = imageOpts.height
        canvas.style = "margin: 4px width:${imageOpts.width}px height:${imageOpts.height}px"
        container.appendChild(canvas)
        tf.browser.toPixels(resized, canvas).await
        resized.dispose()
    } // Render a table of images, we will show the output for each filter
    // in the convolution.

    private fun TensorflowVisualization.Render.imageTable(container: HTMLElement, headerData, data) {
        var table = d3.select(container).select("table")

        if (table.size() === 0) {
            table = d3.select(container).append("table")
            table.append("thead").append("tr")
            table.append("tbody")
        }

        var headers = table.select("thead").select("tr").selectAll("th").data(headerData)
        var headersEnter = headers.enter().append("th")
        headers.merge(headersEnter).each{d, i, group ->
            var node = group[i]

            if (typeof d == "string") {
                node.innerHTML = d
            } else {
                renderImage(node, d, {
                        width: 25,
                        height: 25
                })
            }
        }
        var rows = table.select("tbody").selectAll("tr").data(data)
        var rowsEnter = rows.enter().append("tr")
        var cells = rows.merge(rowsEnter).selectAll("td").data(d => d)
        var cellsEnter = cells.enter().append("td")
        cells.merge(cellsEnter).each{d, i, group ->
            var node = group[i]
            renderImage(node, d, {
                    width: 40,
                    height: 40
            })
        }
        cells.exit().remove()
        rows.exit().remove()
    }

    fun getActivationTable(layerName) {
        val exampleImageSize = 28
        val layer = model.getLayer(layerName) // Get the filters

        var filters = tf.tidy(() => layer.kernel.`val`.transpose([3, 0, 1, 2]).unstack()) // It is hard to draw high dimensional filters so we just use a string

        if (filters[0].shape[2] > 3) {
            filters = filters.map((d, i) => `Filter ${i + 1}`)
        }

        filters.unshift("Input") // Get the inputs

        var numExamples = examples.xs.shape[0]
        var xs = examples.xs.reshape([numExamples, exampleImageSize, exampleImageSize, 1]) // Get the activations

        var activations = tf.tidy{
            return getActivation(xs, model, layer).unstack()
        }
        var activationImageSize = activations[0].shape[0] // e.g. 24

        var numFilters = activations[0].shape[2] // e.g. 8

        var filterActivations = activations.map((activation, i) => {
            // activation has shape [activationImageSize, activationImageSize, i]
            var unpackedActivations = Array(numFilters).fill(0).map((_, i) => activation.slice([0, 0, i], [activationImageSize, activationImageSize, 1])) // prepend the input image

            var inputExample = tf.tidy(() => xs.slice([i], [1]).reshape([exampleImageSize, exampleImageSize, 1]))
            unpackedActivations.unshift(inputExample)
            return unpackedActivations
        })
        return {
                filters,
                filterActivations
        }
    }
    */

    /**
     *
     * */
    private class TrainingResult(
        val input: Array<Array<Json>>,
        val output: Array<Array<Json>>,
        val label: Array<Array<Json>>
    )

    /**
     *
     * */
    private fun TensorflowVisualization.Render.imageTable(container: HTMLElement, data: List<TrainingResult>) {
        fun addRow(vararg colContents: HTMLElement?): HTMLTableRowElement{
            val row = document.createElement("tr") as HTMLTableRowElement
            colContents.forEach {
                val col = document.createElement("td") as HTMLTableCellElement
                col.append(it)
                row.append(col)
            }
            return row
        }
        fun String.getDiv(): HTMLDivElement{
            val div = document.createElement("div") as HTMLDivElement
            div.innerText = this
            return div
        }
        fun Glyph?.getCanvas(): HTMLCanvasElement?{
            val canvas = document.createElement("canvas") as HTMLCanvasElement
            canvas.width = 80
            canvas.height = 80
            val ctx = canvas.getContext("2d") as? CanvasRenderingContext2D?: return null
            ctx.clearRect(0.0, 0.0, canvas.width.toDouble(), canvas.height.toDouble())
            val x = 10
            val y = 60
            val fontSize = 50
            this?.draw(ctx, x, y, fontSize)
            return canvas
        }
        val table = document.createElement("table") as HTMLTableElement
        table.append(addRow(
            "InputRefC".getDiv(),
            "InputRefG".getDiv(),
            "Input".getDiv(),
            "Output".getDiv(),
            "Label".getDiv()
        ))
        data.forEach {
            table.append(addRow(
                it.input.getOrNull(1)?.toGlyph()?.getCanvas(),
                it.input.getOrNull(2)?.toGlyph()?.getCanvas(),
                it.input.getOrNull(0)?.toGlyph()?.getCanvas(),
                it.output.getOrNull(0)?.toGlyph()?.getCanvas(),
                it.label.getOrNull(0)?.toGlyph()?.getCanvas()
            ))
        }
        container.innerHTML = ""
        container.append(table)
    }

    /**
     * 每次訓練的樣本數
     * */
    var batchSize = 32

    /**
    * 訓練多少代
    * */
    var epochs = 100

    /**
     *
     * */
    private fun trainingArgs(inputs: Array<Array<Array<Json>>>, labels: Array<Array<Array<Json>>>, inputsTensor: Tensor): Json {
        return jsObject{
            this.batchSize = batchSize
            this.epochs = epochs
            shuffle = true
            callbacks = jsObject {
                val fitCallbacks = tfvis.show.fitCallbacks(
                    jsObject{ tab = "Training"; name = "Training Performance"; },
                    arrayOf("loss", "mse"),
                    jsObject{ height = 200 }
                )
                //onTrainBegin = fun(logs: dynamic){fitCallbacks.onTrainBegin(logs)}
                //onTrainEnd = fun(logs: dynamic){fitCallbacks.onTrainEnd(logs)}
                //onEpochBegin = fun(epoch: dynamic, logs: dynamic){fitCallbacks.onEpochBegin(epoch, logs)}
                onEpochEnd = fun(epoch: dynamic, logs: dynamic) {fitCallbacks.onEpochEnd(batch, logs)}
                //onBatchBegin = fun(batch: dynamic, logs: dynamic){fitCallbacks.onBatchBegin(batch, logs)}
                onBatchEnd = fun(batch: dynamic, logs: dynamic) {fitCallbacks.onBatchEnd(epoch, logs)
                    //預測輸出
                    val outputsTensor = modelManager.predict(inputsTensor)!!//////////////////////////////////////
                    val outputs = dataConverter.decodeTensor(outputsTensor)
                    //集合數據
                    val trainingResults = inputs.mapIndexed { index, input ->
                        TrainingResult(input, outputs[index], labels[index])
                    }
                    //顯示預測
                    val container = tfvis.visor().surface(jsObject{ tab = "Training"; name = "感受字感"; }).drawArea
                    tfvis.render.imageTable(container, trainingResults)
                }
                //onYield = fun(epoch: dynamic, batch: dynamic, logs: dynamic){fitCallbacks.onYield(epoch, batch, logs)}
            }
        }
    }

    /**
     * 訓練
     * */
    fun train(){
        getTrainingData{ inputs: Array<Array<Array<Json>>>, labels: Array<Array<Array<Json>>> ->
            //轉成Tensor
            val inputsTensor = dataConverter.encodeTensor(inputs)
            val labelsTensor = dataConverter.encodeTensor(labels)
            //訓練
            modelManager.train(inputsTensor, labelsTensor, trainingArgs(inputs, labels, inputsTensor))?.await()
        }
    }

    /**
     * 預測
     * */
    fun predict(vararg glyphUnicodes: Int){
        getPredictingData(glyphUnicodes.toTypedArray()){ input: Array<Array<Array<Json>>>, unicode: Int ->
            //轉成Tensor
            val inputTensor = dataConverter.encodeTensor(input)
            //預測輸出
            val outputTensor = modelManager.predict(inputTensor)?:return@getPredictingData
            val output = dataConverter.decodeTensor(outputTensor)
            //集合數據
            val trainingResults = arrayOf(TrainingResult(input[0], output[0], arrayOf())).toList()
            //顯示預測
            val container = tfvis.visor().surface(jsObject{ tab = "Training"; name = "預測結果"; }).drawArea
            tfvis.render.imageTable(container, trainingResults)
        }
    }

    /**
     * 寫書法
     * */
    fun write(vararg glyphUnicodes: Int){
        getPredictingData(glyphUnicodes.toTypedArray()){ input: Array<Array<Array<Json>>>, unicode: Int ->
            //轉成Tensor
            val inputTensor = dataConverter.encodeTensor(input)
            //製作字形
            val outputTensor = modelManager.predict(inputTensor)?:return@getPredictingData
            val output = dataConverter.decodeTensor(outputTensor)
            //包裝成Path
            val path = Path()
            path.commands = output[0][0]
            //包裝成Glyph
            val glyph = output[0][0].toGlyph(unicode)
            //將新成字形加入到Font
            font.getGlyphByUnicode(unicode)?.reservePaths?.add(path)?:font.add(glyph)
        }
    }
}