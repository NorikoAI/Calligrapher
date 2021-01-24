package org.sourcekey.NorikoAI.Calligrapher

import ExtendedFun.*
import OpentypeJS.Font
import OpentypeJS.Glyph
import OpentypeJS.Path
import kotlinext.js.jsObject
import kotlin.js.Json


/**
 *
 * */
fun Path.setCommands(svgPathData: String): Path{

    fun String.insert(index: Int, string: String): String {
        if (index > 0) {
            return this.substring(0, index) + string + this.substring(index, this.length)
        }
        return string + this
    }

    fun Char.isTypeChar(): Boolean{
        val typeChars = "MmLlCcQqZz"
        for(typeChar in typeChars){
            if(typeChar == this){return true}
        }
        return false
    }

    fun String.addSpaceOfNeedSplitSvgPathData(): String{
        var string = this
        var i = 0
        while (i < string.length){
            if(string[i].isTypeChar()){
                string = string.insert(i + 1, " ").insert(i, " ")
                i += 2
            }else if(string[i] == '-'){
                string = string.insert(i, " ")
                i += 1
            }
            i++
        }
        return string
    }

    fun Array<String>.getTypeCharLocation(): Array<Int>{
        val typeCharLocation = ArrayList<Int>()
        this.forEachIndexed{index, svgPathElement ->
            if(svgPathElement[0].isTypeChar()){
                typeCharLocation.add(index)
            }
        }
        return typeCharLocation.toTypedArray()
    }

    fun Char.getTypeParameterOfNumber(): Int?{
        when(this){
            'M' -> return 2
            'm' -> return 2
            'L' -> return 2
            'l' -> return 2
            'C' -> return 6
            'c' -> return 6
            'Q' -> return 4
            'q' -> return 4
            'Z' -> return 0
            'z' -> return 0
            else -> return null
        }
    }

    fun isTypeParameterNeat(type: Char, start: Int, stop: Int): Boolean{
        val number = type.getTypeParameterOfNumber()
        return number?:0 <= stop-start
    }

    fun Path.setCommand(svgPathElements: Array<String>, _type: Char, start: Int, stop: Int){
        val svgPathElements = svgPathElements
        if(isTypeParameterNeat(_type, start, stop)){
            if (_type == 'M' || _type == 'm') {
                this.moveTo(
                        svgPathElements[start + 1].toInt(),
                        svgPathElements[start + 2].toInt()
                )
            } else if (_type == 'L' || _type == 'l') {
                this.lineTo(
                        svgPathElements[start + 1].toInt(),
                        svgPathElements[start + 2].toInt()
                )
            } else if (_type == 'C' || _type == 'c') {
                this.curveTo(
                        svgPathElements[start + 1].toInt(),
                        svgPathElements[start + 2].toInt(),
                        svgPathElements[start + 3].toInt(),
                        svgPathElements[start + 4].toInt(),
                        svgPathElements[start + 5].toInt(),
                        svgPathElements[start + 6].toInt()
                )
            } else if (_type == 'Q' || _type == 'q') {
                this.quadTo(
                        svgPathElements[start + 1].toInt(),
                        svgPathElements[start + 2].toInt(),
                        svgPathElements[start + 3].toInt(),
                        svgPathElements[start + 4].toInt()
                )
            } else if (_type == 'Z' || _type == 'z') {
                this.close()
            }
        }
    }

    fun Path.setCommands(svgPathElements: Array<String>, typeCharLocations: Array<Int>){
        var i = 0
        while (i < typeCharLocations.size){
            val type = svgPathElements[typeCharLocations[i]][0]
            val start = typeCharLocations[i]
            val stop = (typeCharLocations.getOrNull(i+1)?:svgPathElements.lastIndex)-1
            this.setCommand(svgPathElements, type, start, stop)
            i++
        }
    }

    val svgPathElements = svgPathData.addSpaceOfNeedSplitSvgPathData().split(" ").toTypedArray()
    val typeCharLocations = svgPathElements.getTypeCharLocation()
    this.setCommands(svgPathElements, typeCharLocations)
    return this
}

/**
 *
 * */
val Glyph.reservePaths: ArrayLinkList<Path> get(){
    return asDynamic().reservePaths as? ArrayLinkList<Path> ?: run {
        val paths = ArrayLinkList<Path>()
        paths.onAddListener = fun(path: Path){ paths.designated(path) }
        paths.onSetListener = fun(path: Path){ paths.designated(path) }
        paths.onRemoveListener = fun(path: Path){ paths.designated(0) }
        paths.addOnNodeEventListener(object : ArrayLinkList.OnNodeEventListener<Path> {
            override fun onNodeChanged(
                preChangeNodeID: Int?, postChangeNodeID: Int?,
                preChangeNode: Path?, postChangeNode: Path?
            ) { this@reservePaths.path = postChangeNode?: return }
        })
        paths.add(path?: return paths)
        asDynamic().reservePaths = paths
        asDynamic().reservePaths as ArrayLinkList<Path>
    }
}

/**
 *
 * */
operator fun Font.set(index: Int, glyph: Glyph) = glyphs.push(index, fun(): Glyph{return glyph})

/**
 *
 * */
operator fun Font.get(index: Int): Glyph? = glyphs.get(index)

/**
 *
 * */
fun Font.size(): Int = glyphs.length.toInt()

/**
 *
 * */
fun Font.add(glyph: Glyph) = set(size(), glyph)

/**
 *
 * */
operator fun Font.iterator(): Iterator<Glyph?> = object : Iterator<Glyph?>{
    var i = 0
    override fun hasNext(): Boolean = i < size()
    override fun next(): Glyph? = get(i++)
}

/**
 * 記低上次call Font.getOrNullGlyphByUnicode()搵到嘅Index
 *
 * 為Font.getOrNullGlyphByUnicode()更有效搵Glyph
 * 以上次搵到嘅Index開始起再搵
 * 因多數情況下重複call呢個function都係順住數字咁call
 * */
private var beforeFontGetOrNullGlyphByUnicodeIndex = 0

/**
 *
 * */
fun Font.getGlyphByUnicode(unicode: Int): Glyph?{
    return charToGlyph(String.fromCodePoint(unicode))
    /*return Object.values<Glyph>(this.glyphs.glyphs).findFromIndexStart(
        beforeFontGetOrNullGlyphByUnicodeIndex,
        fun(index, glyph): Boolean{
            beforeFontGetOrNullGlyphByUnicodeIndex = index
            return glyph.unicode == unicode
        }
    )*/
}

/**
 * 係米已選擇保留嘅字形
 * 保留字形可避免再更改此字形
 * */
var Glyph.isKeep: Boolean
    get() = asDynamic().isKeep as? Boolean ?: false
    set(value){ asDynamic().isKeep = value }

/**
 * 過濾未成為保留嘅字形
 *
 * @return 係保留嘅字形就出Glyph/唔係保留嘅字形就出null
 * */
fun Glyph.filterNotKeep(): Glyph?{
    if(isKeep){ return this }
    return null
}

/**
 * 過濾無效字形
 *
 * @return 字形有效出Glyph/無效字形出null
 * */
fun Glyph.filterInvalid(): Glyph?{
    if(path.commands.isNotEmpty() && unicode != null && index != 0){ return this }
    return null
}

/**
 *
 * */
fun Font.clearNullGlyph(): Font{
    for(glyph in this){
        if(glyph?.path?.commands?.isEmpty()?:true || glyph?.unicode == null){
            //Object.delete(glyphs, property)
        }
    }
    return this
}

fun Path.toGlyph(unicode: Int = 0): Glyph{
    //包裝成Glyph
    return Glyph(jsObject{
        this.unicode = unicode
        this.path = this@toGlyph
    })
}

fun Array<Json>.toGlyph(unicode: Int = 0): Glyph{
    //包裝成Path
    val path = Path()
    path.commands = this
    //包裝成Glyph
    return path.toGlyph(unicode)
}
