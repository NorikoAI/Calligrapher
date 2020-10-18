package org.sourcekey.NorikoAI.Calligrapher


import kotlinext.js.*
import kotlinx.html.SVG
import logo.kotlinLogo
import logo.reactLogo
import org.w3c.dom.RenderingContext
import react.RBuilder
import react.dom.*
import kotlin.js.Promise
import kotlin.js.Json


abstract class ObjectGlyph: JsObject

fun ObjectGlyph.toArray(): Array<OpentypeJS.Glyph> = Object.values(this)

@JsModule("opentype.js")
external object OpentypeJS{
    open class BoundingBox

    open class Path{

        var commands: Array<Json>

        var fill: String

        var stroke: String?

        var strokeWidth: Int

        fun draw(ctx: RenderingContext)

        fun getBoundingBox(): BoundingBox

        fun toPathData(decimalPlaces: Int = definedExternally): String

        fun toSVG(decimalPlaces: Int = definedExternally): String

        fun moveTo(x: Int, y: Int)

        fun lineTo(x: Int, y: Int)

        fun curveTo(x1: Int, y1: Int, x2: Int, y2: Int, x: Int, y: Int)

        fun quadTo(x1: Int, y1: Int, x: Int, y: Int)

        fun close()
    }

    open class Glyph(options: Json){

        val advanceWidth: Int

        val index: Int

        val leftSideBearing: Int

        val name: String

        val unicode: Int?

        val unicodes: Array<Int?>

        val font: Font

        val path: Path

        val xMin: Int

        val yMin: Int

        val xMax: Int

        val yMax: Int

        fun getPath(x: Int, y: Int, fontSize: Int):dynamic

        fun getBoundingBox(): BoundingBox

        fun draw(ctx: RenderingContext, text: String, x: Int, y: Int, fontSize: Int, options: Json)

        fun drawPoints(ctx: RenderingContext, text: String, x: Int, y: Int, fontSize: Int, options: Json)

        fun drawMetrics(ctx: RenderingContext, text: String, x: Int, y: Int, fontSize: Int, options: Json)
    }

    open class Font(options: Json){

        val ascender: Int

        val cffEncoding: dynamic//

        val defaultWidthX: Int

        val descender: Int

        val encoding: dynamic//

        val glyphNames: dynamic//

        open class GlyphSet{
            val font: Font

            val glyphs: ObjectGlyph
        }

        val glyphs: GlyphSet

        val gsubrs: Array<dynamic>//

        val gsubrsBias: Int

        val isCIDFont: Boolean

        val kerningPairs: Array<dynamic>//

        val nGlyphs: Int

        val names: dynamic//

        val nominalWidthX: Int

        val numGlyphs: Int

        val numberOfHMetrics: Int

        val outlinesFormat: String

        val position: dynamic//

        val subrs: Array<dynamic>//

        val subrsBias: Int

        val substitution: dynamic//

        val supported: Boolean

        val tables: dynamic//

        val unitsPerEm: Int

        fun getPath(text: String, x: Int = definedExternally, y: Int = definedExternally, fontSize: Int = definedExternally, options: Json = definedExternally): Path

        fun draw(ctx: RenderingContext, text: String, x: Int, y: Int, fontSize: Int, options: Json)

        fun drawPoints(ctx: RenderingContext, text: String, x: Int, y: Int, fontSize: Int, options: Json)

        fun drawMetrics(ctx: RenderingContext, text: String, x: Int, y: Int, fontSize: Int, options: Json)

        fun stringToGlyphs(string: String): dynamic

        fun charToGlyph(char: Char): Glyph

        fun getKerningValue(leftGlyph: Int, rightGlyph: Int): Int

        fun getAdvanceWidth(text: String, fontSize: Int, options: Json): String

        fun download()
    }

    fun load(url: String): Promise<Font>

    fun load(url: String, callback: (err: dynamic, font: Font)->Unit): Promise<Font>

    fun loadSync(url: String): Font

    fun parse(buffer: dynamic): Promise<Font>
}

fun OpentypeJS.Path.setCommands(svgPathData: String): OpentypeJS.Path{

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

    fun OpentypeJS.Path.setCommand(svgPathElements: Array<String>, _type: Char, start: Int, stop: Int){
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

    fun OpentypeJS.Path.setCommands(svgPathElements: Array<String>, typeCharLocations: Array<Int>){
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

fun RBuilder.glyph(glyph: OpentypeJS.Glyph){
    div("glyph") {
        div {
            +glyph.unicode.toString()
        }
        svg {
            +glyph.path.toSVG()
        }
        div {
            +glyph.unicode.toString()
        }
    }
}