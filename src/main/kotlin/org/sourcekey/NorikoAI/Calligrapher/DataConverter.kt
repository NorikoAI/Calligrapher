package org.sourcekey.NorikoAI.Calligrapher

import ExtendedFun.equals
import ExtendedFun.values
import OpentypeJS.Glyph
import TensorFlowJS.Tensor
import TensorFlowJS.tf
import kotlinext.js.Object
import kotlinext.js.jsObject
import kotlin.js.Json
import kotlin.math.roundToInt



class DataConverter {

    private abstract class ConverterDimension{

        companion object{
            /**
             *
             * */
            protected fun Array<Int>.selfMul(): Int{
                var number = 1
                this.map { number *= it }
                return number
            }

            /**
             *
             *
            private fun Array<Int>.selfMulReverse(number: Int): Int{
            var downToCount = number
            val array = ArrayList<Int>()
            for(i in this.last() downTo 0){
            array.add(this.getOrNull(i)?:return array.toTypedArray().selfMul())
            downToCount--
            if(downToCount <= 0){return array.toTypedArray().selfMul()}
            }
            return array.toTypedArray().selfMul()
            }*/

            /**
             * 記錄字形輪廓線嘅字串長度
             *
             * 因TensorflowJS需要每次輸入嘅數據形狀必要一樣
             * 所以要統一所有字串長度
             * **此值必須大於最長字形輪廓線嘅字串長度**
             */
            protected var glyphPathCommandLength = 256

            /**
             *
             * */
            val dataShape = arrayOf(3, glyphPathCommandLength, 12)

            /**
             *
             * */
            protected val max = 1024

            /**
             *
             * */
            protected val min = -1024

            /**
             *
             * */
            protected val tensorMax = tf.tensor(arrayOf(max)).max()

            /**
             *
             * */
            protected val tensorMin = tf.tensor(arrayOf(min)).min()

            /**
             *
             * */
            protected enum class Type{
                NULL, M, L, C, Q, Z;

                companion object{

                    val size: Int
                        get() = values().size

                    val lastIndex: Int
                        get() = values().lastIndex

                    fun valueOfUpperCase(value: String): Type?{
                        return values().find { t -> t.equals(value) }
                    }
                }

                fun equals(string: String): Boolean {
                    return name.toUpperCase() == string.toUpperCase()
                }

                fun equals(char: Char): Boolean {
                    return equals(char.toString())
                }
            }

            /**
             *
             * */
            protected fun Json.commandDecode():Array<Double>?{
                val doubleArray = ArrayList<Double>()
                val commandParameters = Object.values<Glyph>(this)
                val typeString = commandParameters.getOrNull(0) as? String?:return null
                val type = Type.valueOfUpperCase(typeString)?:Type.NULL
                Type.values().forEach { t -> doubleArray.add(if(type.equals(t)){max.toDouble()}else{0.0})}
                type.equals()
                doubleArray.add((commandParameters.getOrNull(1) as? Double)?:if(type.equals(Type.C, Type.Q, Type.M, Type.L)){return null}else{0.0})
                doubleArray.add((commandParameters.getOrNull(2) as? Double)?:if(type.equals(Type.C, Type.Q, Type.M, Type.L)){return null}else{0.0})
                doubleArray.add((commandParameters.getOrNull(3) as? Double)?:if(type.equals(Type.C, Type.Q)){return null}else{0.0})
                doubleArray.add((commandParameters.getOrNull(4) as? Double)?:if(type.equals(Type.C, Type.Q)){return null}else{0.0})
                doubleArray.add((commandParameters.getOrNull(5) as? Double)?:if(type.equals(Type.C)){return null}else{0.0})
                doubleArray.add((commandParameters.getOrNull(6) as? Double)?:if(type.equals(Type.C)){return null}else{0.0})
                return doubleArray.toTypedArray()
            }

            /**
             *
             * */
            protected fun Array<Double>.commandEncode(): Json?{
                val types = this.sliceArray(IntRange(0, Type.lastIndex))
                fun Array<Double>.maxElementIndex(): Int{
                    var maxIndex = 0
                    var maxIndexValue = 0.0
                    this.forEachIndexed { index, d -> if(maxIndexValue < d){
                        maxIndex = index
                        maxIndexValue = d
                    }}
                    return maxIndex
                }
                val typesMax = types.maxElementIndex()
                val type = Type.values().getOrNull(typesMax)
                when(type){
                    Type.M -> { return jsObject {
                        asDynamic().type = "M"
                        asDynamic().x = this@commandEncode.getOrNull(Type.lastIndex+1)?.roundToInt()?:return null
                        asDynamic().y = this@commandEncode.getOrNull(Type.lastIndex+2)?.roundToInt()?:return null
                    } }
                    Type.L -> { return jsObject {
                        asDynamic().type = "L"
                        asDynamic().x = this@commandEncode.getOrNull(Type.lastIndex+1)?.roundToInt()?:return null
                        asDynamic().y = this@commandEncode.getOrNull(Type.lastIndex+2)?.roundToInt()?:return null
                    } }
                    Type.C -> { return jsObject {
                        asDynamic().type = "C"
                        asDynamic().x1 = this@commandEncode.getOrNull(Type.lastIndex+1)?.roundToInt()?:return null
                        asDynamic().y1 = this@commandEncode.getOrNull(Type.lastIndex+2)?.roundToInt()?:return null
                        asDynamic().x2 = this@commandEncode.getOrNull(Type.lastIndex+3)?.roundToInt()?:return null
                        asDynamic().y2 = this@commandEncode.getOrNull(Type.lastIndex+4)?.roundToInt()?:return null
                        asDynamic().x = this@commandEncode.getOrNull(Type.lastIndex+5)?.roundToInt()?:return null
                        asDynamic().y = this@commandEncode.getOrNull(Type.lastIndex+6)?.roundToInt()?:return null
                    } }
                    Type.Q -> { return jsObject {
                        asDynamic().type = "Q"
                        asDynamic().x1 = this@commandEncode.getOrNull(Type.lastIndex+1)?.roundToInt()?:return null
                        asDynamic().y1 = this@commandEncode.getOrNull(Type.lastIndex+2)?.roundToInt()?:return null
                        asDynamic().x = this@commandEncode.getOrNull(Type.lastIndex+3)?.roundToInt()?:return null
                        asDynamic().y = this@commandEncode.getOrNull(Type.lastIndex+4)?.roundToInt()?:return null
                    } }
                    Type.Z -> { return jsObject {
                        asDynamic().type = "Z"
                    } }
                }
                return null
            }
        }

        /**
         *
         * */
        abstract val inputShape: Array<Int>

        /**
         *
         * */
        abstract val outputUnits: Int

        /**
         * 把資料轉成Tensor
         * */
        abstract val encodeTensor: (data: Array<Array<Array<Json>>>) -> Tensor

        /**
         * 把Tensor轉成資料
         *
         * @param
         * @returns
         */
        abstract val decodeTensor: (tensor: Tensor) -> Array<Array<Array<Json>>>
    }

    private val converterDimension_1D = object : ConverterDimension() {

        override val inputShape: Array<Int>
            get() = arrayOf(dataShape[2] * dataShape[1] * dataShape[0])//arrayOf(dataShape.selfMul())

        override val outputUnits: Int
            get() = dataShape[2] * dataShape[1] * dataShape[0]//dataShape.selfMulReverse(3)

        /**
         *
         * */
        private fun Array<Json>.uniformLength(): Array<Double>{
            val array = ArrayList<Double>()
            var i = 0
            while(i < dataShape[1] ){
                val command = this.getOrNull(i)?.commandDecode()?:run{
                    val array = ArrayList<Double>()
                    val type = Companion.Type.NULL.ordinal.toDouble()
                    Companion.Type.values().forEach { t -> array.add(if(type.equals(t)){max.toDouble()}else{0.0})}
                    array.addAll(arrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    array.toTypedArray()
                }
                array.addAll(command)
                i++
            }
            return array.toTypedArray()
        }

        /**
         *
         * */
        private fun Array<Double>.ununiformLength(): Array<Json>{
            val array = ArrayList<Json>()
            for(i in this.indices step dataShape[2]){
                val command = this.sliceArray(IntRange(i, i + dataShape[2] - 1)).commandEncode()
                if(command != null){ array.add(command) }
            }
            return array.toTypedArray()
        }

        override val encodeTensor: (data: Array<Array<Array<Json>>>) -> Tensor
            get() = fun(data: Array<Array<Array<Json>>>): Tensor{
                // 使用tf.tidy讓除了回傳值以外，中間過程中的所佔用的空間釋放掉
                return tf.tidy{
                    // 統一資料長度 同 轉成數字
                    val toIntedData = data.map{d ->
                        val array = ArrayList<Double>()
                        d.forEach { array.addAll(it.uniformLength()) }
                        array.toTypedArray()
                    }.toTypedArray()
                    //轉成Tensor
                    val tensor = tf.tensor2d(toIntedData)!!
                    //正規化 將 (tensor內的資料-最小值)/(最大值-最小值)) 出來的結果在0-1之間
                    return@tidy tensor.sub(tensorMin).div(tensorMax.sub(tensorMin))
                }
            }

        override val decodeTensor: (tensor: Tensor) -> Array<Array<Array<Json>>>
            get() = fun(tensor: Tensor): Array<Array<Array<Json>>>{
                return tf.tidy{
                    //轉回原本的數= 數字*(最大值-最小值)+最小值
                    val orgTenser = tensor.mul(tensorMax.sub(tensorMin)).add(tensorMin)
                    //轉返普通Array
                    val data = orgTenser.arraySync() as Array<Array<Double>>
                    //轉返字串　同　還原資料長度
                    return@tidy data.map{d ->
                        val size = dataShape[1] * dataShape[2]
                        val array = ArrayList<Array<Json>>()
                        for(i in d.indices step size){
                            val command = d.sliceArray(IntRange(i, i + size - 1))
                            array.add(command.ununiformLength())
                        }
                        array.toTypedArray()
                    }.toTypedArray()
                }
            }

    }

    private val converterDimension_2D = object : ConverterDimension() {

        override val inputShape: Array<Int>
            get() = arrayOf(dataShape[0], dataShape[1] * dataShape[2])

        override val outputUnits: Int
            get() = dataShape[2] * dataShape[1]

        /**
         *
         * */
        private fun Array<Json>.uniformLength(): Array<Double>{
            val array = ArrayList<Double>()
            var i = 0
            while(i < dataShape[1] ){
                val command = this.getOrNull(i)?.commandDecode()?:run{
                    val array = ArrayList<Double>()
                    val type = Companion.Type.NULL.ordinal.toDouble()
                    Companion.Type.values().forEach { t -> array.add(if(type.equals(t)){max.toDouble()}else{0.0})}
                    array.addAll(arrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    array.toTypedArray()
                }
                array.addAll(command)
                i++
            }
            return array.toTypedArray()
        }

        /**
         *
         * */
        private fun Array<Double>.ununiformLength(): Array<Json>{
            val array = ArrayList<Json>()
            for(i in this.indices step dataShape[2]){
                val command = this.sliceArray(IntRange(i, i + dataShape[2] - 1)).commandEncode()
                if(command != null){ array.add(command) }
            }
            return array.toTypedArray()
        }

        override val encodeTensor: (data: Array<Array<Array<Json>>>) -> Tensor
            get() = fun(data: Array<Array<Array<Json>>>): Tensor{
                // 使用tf.tidy讓除了回傳值以外，中間過程中的所佔用的空間釋放掉
                return tf.tidy{
                    // 統一資料長度 同 轉成數字
                    val toIntedData = data.map{d -> d.map { it.uniformLength() }.toTypedArray() }.toTypedArray()
                    //轉成Tensor
                    val tensor = tf.tensor3d(toIntedData)!!
                    //正規化 將 (tensor內的資料-最小值)/(最大值-最小值)) 出來的結果在0-1之間
                    return@tidy tensor.sub(tensorMin).div(tensorMax.sub(tensorMin))
                }
            }

        override val decodeTensor: (tensor: Tensor) -> Array<Array<Array<Json>>>
            get() = fun(tensor: Tensor): Array<Array<Array<Json>>>{
                return tf.tidy{
                    //轉回原本的數= 數字*(最大值-最小值)+最小值
                    val orgTenser = tensor.mul(tensorMax.sub(tensorMin)).add(tensorMin)
                    //轉返普通Array
                    val data = orgTenser.arraySync() as Array<Array<Array<Double>>>
                    //轉返字串　同　還原資料長度
                    return@tidy data.map{d -> d.map { it.ununiformLength() }.toTypedArray() }.toTypedArray()
                }
            }
    }

    private val converterDimension_3D = object : ConverterDimension() {

        override val inputShape: Array<Int>
            get() = dataShape

        override val outputUnits: Int
            get() = dataShape[2]

        /**
         *
         * */
        private fun Array<Json>.uniformLength(): Array<Array<Double>>{
            val array = ArrayList<Array<Double>>()
            var i = 0
            while(i < dataShape[1] ){
                val command = this.getOrNull(i)?.commandDecode()?:run{
                    val arrayD = ArrayList<Double>()
                    val type = Companion.Type.NULL.ordinal.toDouble()
                    Companion.Type.values().forEach { t -> arrayD.add(if(type.equals(t)){max.toDouble()}else{0.0})}
                    arrayD.addAll(arrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    arrayD.toTypedArray()
                }
                array.add(command)
                i++
            }
            return array.toTypedArray()
        }

        /**
         *
         * */
        private fun Array<Array<Double>>.ununiformLength(): Array<Json>{
            val array = ArrayList<Json>()
            for(element in this){
                val command = element.commandEncode()
                if(command != null){ array.add(command) }
            }
            return array.toTypedArray()
        }

        override val encodeTensor: (data: Array<Array<Array<Json>>>) -> Tensor
            get() = fun(data: Array<Array<Array<Json>>>): Tensor{
                // 使用tf.tidy讓除了回傳值以外，中間過程中的所佔用的空間釋放掉
                return tf.tidy{
                    // 統一資料長度 同 轉成數字
                    val toIntedData = data.map{d -> d.map { it.uniformLength() }.toTypedArray() }.toTypedArray()
                    //轉成Tensor
                    val tensor = tf.tensor4d(toIntedData)!!
                    //正規化 將 (tensor內的資料-最小值)/(最大值-最小值)) 出來的結果在0-1之間
                    return@tidy tensor.sub(tensorMin).div(tensorMax.sub(tensorMin))
                }
            }

        override val decodeTensor: (tensor: Tensor) -> Array<Array<Array<Json>>>
            get() = fun(tensor: Tensor): Array<Array<Array<Json>>>{
                return tf.tidy{
                    //轉回原本的數= 數字*(最大值-最小值)+最小值
                    val orgTenser = tensor.mul(tensorMax.sub(tensorMin)).add(tensorMin)
                    //轉返普通Array
                    val data = orgTenser.arraySync() as Array<Array<Array<Array<Double>>>>
                    //轉返字串　同　還原資料長度
                    return@tidy data.map{d -> d.map { it.ununiformLength() }.toTypedArray() }.toTypedArray()
                }
            }

    }


    private val converterDimension = converterDimension_2D

    /**
     *
     * */
    val inputShape = converterDimension.inputShape

    /**
     *
     * */
    val outputUnits = converterDimension.outputUnits

    /**
     * 把資料轉成Tensor
     * */
    val encodeTensor = converterDimension.encodeTensor

    /**
     * 把Tensor轉成資料
     *
     * @param
     * @returns
     */
    val decodeTensor = converterDimension.decodeTensor

}