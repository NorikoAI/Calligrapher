package org.sourcekey.NorikoAI.Calligrapher

import kotlinext.js.Object
import kotlinx.html.STYLE
import kotlin.js.Json
import kotlin.math.roundToInt



class DataConverter(val dataShape: Array<Int>) {

    enum class Type{
        NULL, M, L, C, Q, Z;

        companion object{

            val size: Int
                get() = values().size

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

    private val max = 1024

    private val min = -1024

    private val tensorMax = tf.tensor(arrayOf(max)).max()

    private val tensorMin = tf.tensor(arrayOf(min)).min()

    private fun Json.commandDecode():Array<Double>?{
        val doubleArray = ArrayList<Double>()
        val commandParameters = Object.values(this)
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

    private fun Array<Double>.commandEncode(): Json?{
        val types = this.sliceArray(IntRange(0, Type.size-1))
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
            Type.M -> {
                return jsObject {
                    this.type = "M"
                    x = this@commandEncode.getOrNull(Type.size+1)?.roundToInt()?:return null
                    y = this@commandEncode.getOrNull(Type.size+2)?.roundToInt()?:return null
                }
            }
            Type.L -> {
                return jsObject {
                    this.type = "L"
                    x = this@commandEncode.getOrNull(Type.size+1)?.roundToInt()?:return null
                    y = this@commandEncode.getOrNull(Type.size+2)?.roundToInt()?:return null
                }
            }
            Type.C -> {
                return jsObject {
                    this.type = "C"
                    x1 = this@commandEncode.getOrNull(Type.size+1)?.roundToInt()?:return null
                    y1 = this@commandEncode.getOrNull(Type.size+2)?.roundToInt()?:return null
                    x2 = this@commandEncode.getOrNull(Type.size+3)?.roundToInt()?:return null
                    y2 = this@commandEncode.getOrNull(Type.size+4)?.roundToInt()?:return null
                    x = this@commandEncode.getOrNull(Type.size+5)?.roundToInt()?:return null
                    y = this@commandEncode.getOrNull(Type.size+6)?.roundToInt()?:return null
                }
            }
            Type.Q -> {
                return jsObject {
                    this.type = "Q"
                    x1 = this@commandEncode.getOrNull(Type.size+1)?.roundToInt()?:return null
                    y1 = this@commandEncode.getOrNull(Type.size+2)?.roundToInt()?:return null
                    x = this@commandEncode.getOrNull(Type.size+3)?.roundToInt()?:return null
                    y = this@commandEncode.getOrNull(Type.size+4)?.roundToInt()?:return null
                }
            }
            Type.Z -> {
                return jsObject {
                    this.type = "Z"
                }
            }
        }
        return null
    }

    /**
     *
     * */
    private fun Array<Json>.uniformLength(): Array<Array<Double>>{
        val array = ArrayList<Array<Double>>()
        var i = 0
        while(i < dataShape[1] ){
            val command = this.getOrNull(i)?.commandDecode()?:arrayOf(
                    Type.NULL.ordinal.toDouble(),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
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

    /**
     * 把資料轉成Tensor
     * */
    fun encodeTensor(data: Array<Array<Array<Json>>>): Tensor{
        // 使用tf.tidy讓除了回傳值以外，中間過程中的所佔用的空間釋放掉
        return tf.tidy(fun(): Tensor {
            // 統一資料長度 同 轉成數字
            val toIntedData = data.map{d ->
                arrayOf(
                        d[0].uniformLength(),
                        d[1].uniformLength(),
                        d[2].uniformLength()
                )
            }.toTypedArray()
            //轉成Tensor
            val tensor = tf.tensor4d(toIntedData)!!
            //正規化 將 (tensor內的資料-最小值)/(最大值-最小值)) 出來的結果在0-1之間
            return tensor.sub(tensorMin).div(tensorMax.sub(tensorMin))
        })
    }

    /**
     * Convert a 2D tensor into a string with the CharacterTable's vocabulary.
     *
     * @param x Input 2D tensor.
     * @param calcArgmax Whether to perform `argMax` operation on `x` before
     *   indexing into the `CharacterTable`'s vocabulary.
     * @returns The decoded string.
     */
    fun decodeTensor(tensor: Tensor): Array<Array<Array<Json>>>{
        return tf.tidy(fun(): Array<Array<Array<Json>>>{
            //轉回原本的數= 數字*(最大值-最小值)+最小值
            val orgTenser = tensor.mul(tensorMax.sub(tensorMin)).add(tensorMin)
            //轉返普通Array
            val data = orgTenser.arraySync() as Array<Array<Array<Array<Double>>>>
            //轉返字串　同　還原資料長度
            return data.map{d ->
                arrayOf(
                        d[0].ununiformLength(),
                        d[1].ununiformLength(),
                        d[2].ununiformLength()
                )
            }.toTypedArray()
        })
    }

}