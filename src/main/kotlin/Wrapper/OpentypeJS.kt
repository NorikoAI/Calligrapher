@file:Suppress("INTERFACE_WITH_SUPERCLASS", "OVERRIDING_FINAL_MEMBER", "RETURN_TYPE_MISMATCH_ON_OVERRIDE", "CONFLICTING_OVERLOADS")

import org.khronos.webgl.ArrayBuffer
import org.w3c.dom.CanvasRenderingContext2D
import org.w3c.dom.svg.SVGPathElement
import kotlin.js.Promise

@JsModule("opentype.js")
external object OpentypeJS{
    interface `T$0` {
        @nativeGetter
        operator fun get(tableName: String): Table?
        @nativeSetter
        operator fun set(tableName: String, value: Table)
    }

    open class Font(options: FontConstructorOptionsBase /* FontConstructorOptionsBase & FontOptionsPartial & `T$3` */) {
        open var names: FontNames
        open var unitsPerEm: Number
        open var ascender: Number
        open var descender: Number
        open var createdTimestamp: Number
        open var tables: `T$0`
        open var supported: Boolean
        open var glyphs: GlyphSet
        open var encoding: Encoding
        open var substitution: Substitution
        open var defaultRenderOptions: RenderOptions
        open fun charToGlyph(c: String): Glyph?
        open fun charToGlyphIndex(s: String): Number
        open fun download(fileName: String = definedExternally)
        open fun draw(ctx: CanvasRenderingContext2D, text: String, x: Number = definedExternally, y: Number = definedExternally, fontSize: Number = definedExternally, options: RenderOptions = definedExternally)
        open fun drawMetrics(ctx: CanvasRenderingContext2D, text: String, x: Number = definedExternally, y: Number = definedExternally, fontSize: Number = definedExternally, options: RenderOptions = definedExternally)
        open fun drawPoints(ctx: CanvasRenderingContext2D, text: String, x: Number = definedExternally, y: Number = definedExternally, fontSize: Number = definedExternally, options: RenderOptions = definedExternally)
        open fun forEachGlyph(text: String, x: Number?, y: Number?, fontSize: Number?, options: RenderOptions?, callback: (glyph: Glyph, x: Number, y: Number, fontSize: Number, options: RenderOptions) -> Unit): Number
        open fun getAdvanceWidth(text: String, fontSize: Number = definedExternally, options: RenderOptions = definedExternally): Number
        open fun getEnglishName(name: String): String
        open fun getKerningValue(leftGlyph: Glyph, rightGlyph: Glyph): Number
        open fun getKerningValue(leftGlyph: Glyph, rightGlyph: Number): Number
        open fun getKerningValue(leftGlyph: Number, rightGlyph: Glyph): Number
        open fun getKerningValue(leftGlyph: Number, rightGlyph: Number): Number
        open fun getPath(text: String, x: Number, y: Number, fontSize: Number, options: RenderOptions = definedExternally): Path
        open fun getPaths(text: String, x: Number, y: Number, fontSize: Number, options: RenderOptions = definedExternally): Array<Path>
        open fun glyphIndexToName(gid: Number): String
        open var glyphNames: GlyphNames
        open fun hasChar(c: String): Boolean
        open var kerningPairs: KerningPairs
        open fun nameToGlyph(name: String): Glyph
        open fun nameToGlyphIndex(name: String): Number
        open var numberOfHMetrics: Number
        open var numGlyphs: Number
        open var outlinesFormat: String
        open fun stringToGlyphs(s: String): Array<Glyph>
        open fun toArrayBuffer(): ArrayBuffer
        open fun toBuffer(): ArrayBuffer
        open fun toTables(): Table
        open fun validate()
    }

    interface FontOptions {
        var empty: Boolean?
            get() = definedExternally
            set(value) = definedExternally
        var familyName: String
        var styleName: String
        var fullName: String?
            get() = definedExternally
            set(value) = definedExternally
        var postScriptName: String?
            get() = definedExternally
            set(value) = definedExternally
        var designer: String?
            get() = definedExternally
            set(value) = definedExternally
        var designerURL: String?
            get() = definedExternally
            set(value) = definedExternally
        var manufacturer: String?
            get() = definedExternally
            set(value) = definedExternally
        var manufacturerURL: String?
            get() = definedExternally
            set(value) = definedExternally
        var license: String?
            get() = definedExternally
            set(value) = definedExternally
        var licenseURL: String?
            get() = definedExternally
            set(value) = definedExternally
        var version: String?
            get() = definedExternally
            set(value) = definedExternally
        var description: String?
            get() = definedExternally
            set(value) = definedExternally
        var copyright: String?
            get() = definedExternally
            set(value) = definedExternally
        var trademark: String?
            get() = definedExternally
            set(value) = definedExternally
        var unitsPerEm: Number
        var ascender: Number
        var descender: Number
        var createdTimestamp: Number
        var weightClass: String?
            get() = definedExternally
            set(value) = definedExternally
        var widthClass: String?
            get() = definedExternally
            set(value) = definedExternally
        var fsSelection: String?
            get() = definedExternally
            set(value) = definedExternally
    }

    interface FontOptionsPartial {
        var empty: Boolean?
            get() = definedExternally
            set(value) = definedExternally
        var familyName: String?
            get() = definedExternally
            set(value) = definedExternally
        var styleName: String?
            get() = definedExternally
            set(value) = definedExternally
        var fullName: String?
            get() = definedExternally
            set(value) = definedExternally
        var postScriptName: String?
            get() = definedExternally
            set(value) = definedExternally
        var designer: String?
            get() = definedExternally
            set(value) = definedExternally
        var designerURL: String?
            get() = definedExternally
            set(value) = definedExternally
        var manufacturer: String?
            get() = definedExternally
            set(value) = definedExternally
        var manufacturerURL: String?
            get() = definedExternally
            set(value) = definedExternally
        var license: String?
            get() = definedExternally
            set(value) = definedExternally
        var licenseURL: String?
            get() = definedExternally
            set(value) = definedExternally
        var version: String?
            get() = definedExternally
            set(value) = definedExternally
        var description: String?
            get() = definedExternally
            set(value) = definedExternally
        var copyright: String?
            get() = definedExternally
            set(value) = definedExternally
        var trademark: String?
            get() = definedExternally
            set(value) = definedExternally
        var unitsPerEm: Number?
            get() = definedExternally
            set(value) = definedExternally
        var ascender: Number?
            get() = definedExternally
            set(value) = definedExternally
        var descender: Number?
            get() = definedExternally
            set(value) = definedExternally
        var createdTimestamp: Number?
            get() = definedExternally
            set(value) = definedExternally
        var weightClass: String?
            get() = definedExternally
            set(value) = definedExternally
        var widthClass: String?
            get() = definedExternally
            set(value) = definedExternally
        var fsSelection: String?
            get() = definedExternally
            set(value) = definedExternally
    }

    interface FontConstructorOptionsBase {
        var familyName: String
        var styleName: String
        var unitsPerEm: Number
        var ascender: Number
        var descender: Number
    }

    interface FontNames {
        var copyright: LocalizedName
        var description: LocalizedName
        var designer: LocalizedName
        var designerURL: LocalizedName
        var fontFamily: LocalizedName
        var fontSubfamily: LocalizedName
        var fullName: LocalizedName
        var license: LocalizedName
        var licenseURL: LocalizedName
        var manufacturer: LocalizedName
        var manufacturerURL: LocalizedName
        var postScriptName: LocalizedName
        var trademark: LocalizedName
        var version: LocalizedName
    }

    interface Table {
        @nativeGetter
        operator fun get(propName: String): Any?
        @nativeSetter
        operator fun set(propName: String, value: Any)
        fun encode(): Array<Number>
        var fields: Array<Field>
        fun sizeOf(): Number
        var tables: Array<Table>
        var tableName: String
    }

    interface KerningPairs {
        @nativeGetter
        operator fun get(pair: String): Number?
        @nativeSetter
        operator fun set(pair: String, value: Number)
    }

    interface LocalizedName {
        @nativeGetter
        operator fun get(lang: String): String?
        @nativeSetter
        operator fun set(lang: String, value: String)
    }

    interface Field {
        var name: String
        var type: String
        var value: Any
    }

    open class Glyph(options: GlyphOptions) {
        open var index: Number
        open var xMin: Any
        open var xMax: Any
        open var yMin: Any
        open var yMax: Any
        open var points: Any
        open var name: String
        open var path: Path /* Path | () -> Path */
        open var unicode: Number?
        open var unicodes: Array<Number?>
        open var advanceWidth: Number
        open fun addUnicode(unicode: Number)
        open fun bindConstructorValues(options: GlyphOptions)
        open fun draw(ctx: CanvasRenderingContext2D, x: Number = definedExternally, y: Number = definedExternally, fontSize: Number = definedExternally, options: RenderOptions = definedExternally)
        open fun drawMetrics(ctx: CanvasRenderingContext2D, x: Number = definedExternally, y: Number = definedExternally, fontSize: Number = definedExternally, options: RenderOptions = definedExternally)
        open fun drawPoints(ctx: CanvasRenderingContext2D, x: Number = definedExternally, y: Number = definedExternally, fontSize: Number = definedExternally, options: RenderOptions = definedExternally)
        open fun getBoundingBox(): BoundingBox
        open fun getContours(): Contour
        open fun getMetrics(): Metrics
        open fun getPath(x: Number = definedExternally, y: Number = definedExternally, fontSize: Number = definedExternally, options: RenderOptions = definedExternally, font: Font = definedExternally): Path
    }

    interface GlyphOptions {
        var advanceWidth: Number?
            get() = definedExternally
            set(value) = definedExternally
        var index: Number?
            get() = definedExternally
            set(value) = definedExternally
        var font: Font?
            get() = definedExternally
            set(value) = definedExternally
        var name: String?
            get() = definedExternally
            set(value) = definedExternally
        var path: Path?
            get() = definedExternally
            set(value) = definedExternally
        var unicode: Number?
            get() = definedExternally
            set(value) = definedExternally
        var unicodes: Array<Number>?
            get() = definedExternally
            set(value) = definedExternally
        var xMax: Number?
            get() = definedExternally
            set(value) = definedExternally
        var xMin: Number?
            get() = definedExternally
            set(value) = definedExternally
        var yMax: Number?
            get() = definedExternally
            set(value) = definedExternally
        var yMin: Number?
            get() = definedExternally
            set(value) = definedExternally
    }

    open class GlyphNames(post: Post) {
        open var names: Any
        open fun glyphIndexToName(gid: Number): String
        open fun nameToGlyphIndex(name: String): Number
    }

    open class GlyphSet {
        open var font: Any
        open var glyphs: Any
        constructor(font: Font, glyphs: Array<Glyph>)
        constructor(font: Font, glyphs: Array<() -> Glyph>)
        open fun get(index: Number): Glyph
        open var length: Number
        open fun push(index: Number, loader: () -> Glyph)
    }

    interface Post {
        var glyphNameIndex: Array<Number>?
            get() = definedExternally
            set(value) = definedExternally
        var isFixedPitch: Number
        var italicAngle: Number
        var maxMemType1: Number
        var minMemType1: Number
        var maxMemType42: Number
        var minMemType42: Number
        var names: Array<String>?
            get() = definedExternally
            set(value) = definedExternally
        var numberOfGlyphs: Number?
            get() = definedExternally
            set(value) = definedExternally
        var offset: Array<Number>?
            get() = definedExternally
            set(value) = definedExternally
        var underlinePosition: Number
        var underlineThickness: Number
        var version: Number
    }

    interface `T$1` {
        @nativeGetter
        operator fun get(key: String): Boolean?
        @nativeSetter
        operator fun set(key: String, value: Boolean)
    }

    interface RenderOptions {
        var script: String?
            get() = definedExternally
            set(value) = definedExternally
        var language: String?
            get() = definedExternally
            set(value) = definedExternally
        var kerning: Boolean?
            get() = definedExternally
            set(value) = definedExternally
        var xScale: Number?
            get() = definedExternally
            set(value) = definedExternally
        var yScale: Number?
            get() = definedExternally
            set(value) = definedExternally
        var features: `T$1`?
            get() = definedExternally
            set(value) = definedExternally
    }

    interface Metrics {
        var leftSideBearing: Number
        var rightSideBearing: Number?
            get() = definedExternally
            set(value) = definedExternally
        var xMax: Number
        var xMin: Number
        var yMax: Number
        var yMin: Number
    }

    interface Point {
        var lastPointOfContour: Boolean?
            get() = definedExternally
            set(value) = definedExternally
    }

    open class Path {
        open var fill: String?
        open var stroke: String?
        open var strokeWidth: Number
        open fun bezierCurveTo(x1: Number, y1: Number, x2: Number, y2: Number, x: Number, y: Number)
        open var close: () -> Unit
        open fun closePath()
        open var commands: Array<dynamic /* `T$4` | `T$5` | `T$6` | `T$7` | `T$8` */>
        open var curveTo: (x1: Number, y1: Number, x2: Number, y2: Number, x: Number, y: Number) -> Unit
        open fun draw(ctx: CanvasRenderingContext2D)
        open fun extend(pathOrCommands: Path)
        open fun extend(pathOrCommands: Array<Any /* `T$4` | `T$5` | `T$6` | `T$7` | `T$8` */>)
        open fun extend(pathOrCommands: BoundingBox)
        open fun getBoundingBox(): BoundingBox
        open fun lineTo(x: Number, y: Number)
        open fun moveTo(x: Number, y: Number)
        open fun quadraticCurveTo(x1: Number, y1: Number, x: Number, y: Number)
        open var quadTo: (x1: Number, y1: Number, x: Number, y: Number) -> Unit
        open fun toDOMElement(decimalPlaces: Number): SVGPathElement
        open fun toPathData(decimalPlaces: Number): String
        open fun toSVG(decimalPlaces: Number): String
        open var unitsPerEm: Number
    }

    open class BoundingBox {
        open var x1: Number
        open var y1: Number
        open var x2: Number
        open var y2: Number
        open fun isEmpty(): Boolean
        open fun addPoint(x: Number, y: Number)
        open fun addX(x: Number)
        open fun addY(y: Number)
        open fun addBezier(x0: Number, y0: Number, x1: Number, y1: Number, x2: Number, y2: Number, x: Number, y: Number)
        open fun addQuad(x0: Number, y0: Number, x1: Number, y1: Number, x: Number, y: Number)
    }

    interface Encoding {
        var charset: String
        fun charToGlyphIndex(c: String): Number
        var font: Font
    }

    fun load(url: String, callback: (error: Any, font: Font) -> Unit)

    fun load(url: String): Promise<Font>

    interface `T$2` {
        var lowMemory: Boolean
    }

    fun loadSync(url: String, opt: `T$2` = definedExternally): Font

    fun parse(buffer: Any): Font
}

typealias Contour = Array<OpentypeJS.Point>

typealias Substitution = (font: OpentypeJS.Font) -> Any