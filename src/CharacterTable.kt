package org.sourcekey.NorikoAI.Calligrapher


/**
 * Constructor of CharacterTable.
 * @param chars A string that contains the characters that can appear
 *   in the input.
 */
class CharacterTable(val chars: CharArray) {

    fun charIndices(char: Char): Int?{
        val index = chars.indexOf(char)
        return if(index != -1){ index }else{ null }
    }

    fun indicesChar(index: Int): Char?{
        return chars.getOrNull(index)
    }

    val size: Int
        get() = this.chars.size
/*
    /**
     * Convert a string into a one-hot encoded tensor.
     *
     * @param str The input string.
     * @param numRows Number of rows of the output tensor.
     * @returns The one-hot encoded 2D tensor.
     * @throws If `str` contains any characters outside the `CharacterTable`'s
     *   vocabulary.
     */
    fun encode(string: String, numRows: Int) {
        var buf = tf.buffer([numRows, this.size])
        for (let i = 0 i < str.length ++i) {
            var char = str[i]
            if (this.charIndices[char] == null) {
                throw new Error(`Unknown character: '${char}'`)
            }
            buf.set(1, i, this.charIndices[char])
        }
        return buf.toTensor().as2D(numRows, this.size)
    }

    fun encodeBatch(strings: Array<String>, numRows: Int) {
        var numExamples = strings.length
        var buf = tf.buffer([numExamples, numRows, this.size])
        for (let n = 0 n < numExamples ++n) {
            var str = strings[n]
            for (let i = 0 i < str.length ++i) {
            var char = str[i]
            if (this.charIndices[char] == null) {
                throw new Error(`Unknown character: '${char}'`)
            }
            buf.set(1, n, i, this.charIndices[char])
        }
        }
        return buf.toTensor().as3D(numExamples, numRows, this.size)
    }

    /**
     * Convert a 2D tensor into a string with the CharacterTable's vocabulary.
     *
     * @param x Input 2D tensor.
     * @param calcArgmax Whether to perform `argMax` operation on `x` before
     *   indexing into the `CharacterTable`'s vocabulary.
     * @returns The decoded string.
     */
    fun decode(x, calcArgmax = true) {
        return tf.tidy(() => {
            if (calcArgmax) {
                x = x.argMax(1)
            }
            var xData = x.dataSync()  // TODO(cais): Performance implication?
            let output = ''
            for (var index of Array.from(xData)) {
            output += this.indicesChar[index]
        }
            return output
        })
    }
 */
}