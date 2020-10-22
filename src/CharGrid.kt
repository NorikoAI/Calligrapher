package org.sourcekey.NorikoAI.Calligrapher

import kotlinx.browser.window
import kotlinx.css.*
import kotlinx.html.weekInput
import org.w3c.dom.HTMLCanvasElement
import org.w3c.dom.RenderingContext
import react.*
import react.dom.*
import react.infinitegrid.GridLayout
import styled.css
import styled.styledCanvas
import styled.styledDiv


interface CharProps : RProps {
    var groupKey: Int
    var key: Int
    var unicode: Int
    var onGetGlyph: (Int, (OpentypeJS.Glyph?)->Unit)->Unit
    var onGetGlyphNow: (Int, (OpentypeJS.Glyph?)->Unit)->Unit
}

class CharState: RState{
    var glyph: OpentypeJS.Glyph? = null
    set(value) {
        field = value?:field
    }
}
/*
fun RBuilder.char(groupKey: Int, key: Int, unicode: Int, char: OpentypeJS.Glyph?): RClass<CharProps> {
    //zconsole.log(char)
    val item = rFunction<CharProps>("Char") { props ->
        props.groupKey = groupKey
        props.key = key
        styledDiv{
            css{
                if(char != null){ background = "#FEFEFE" }else{ background = "#FFF7F7" }
                width = 10.vh
                height = 10.vh
                overflow = Overflow.hidden
            }
            div {
                +"U+${unicode.toString(16).toUpperCase()}"
            }
            svg {
                attrs.jsStyle {
                    width = "250px"
                }
                +(char?.path?.toSVG()?:"")
            }
        }
    }
    item.invoke{}
    return item
}
*/
/*
private val char = functionalComponent<CharProps> { props ->
    styledDiv{
        css{
            if(props.glyph != null){ background = "#FEFEFE" }else{ background = "#FFF7F7" }
            width = 10.vh
            height = 10.vh
            overflow = Overflow.hidden
        }
        div {
            +"U+${props.unicode.toString(16).toUpperCase()}"
        }
        svg {
            attrs.jsStyle {
                width = "250px"
            }
            +(props.glyph?.path?.toSVG()?:"")
        }
    }
}*/
private class Char: RComponent<CharProps, CharState>() {

    override fun CharState.init() {
        glyph = null
    }

    var timerID: Int? = null

    override fun componentDidMount() {
        //
        props.onGetGlyphNow(props.unicode, fun(glyph: OpentypeJS.Glyph?){ setState{ this.glyph = glyph } })
        //
        timerID = window.setInterval({
            props.onGetGlyph(props.unicode, fun(glyph: OpentypeJS.Glyph?){ setState{ this.glyph = glyph } })
        }, 5000)
    }

    override fun componentWillUnmount() {
        window.clearInterval(timerID!!)
    }

    override fun RBuilder.render() {
        styledDiv{
            css{
                if(state.glyph != null){ background = "#FAFAFA" }else{ background = "#EEE" } //"#FFF7F7"
                width = 10.vh
                //height = 10.vh
                overflow = Overflow.hidden
            }
            div {
                +"U+${props.unicode.toString(16).toUpperCase()}"
            }
            styledCanvas {
                attrs{
                    width = "200"
                    height = "200"
                }
                css {
                    width = 10.vh
                    height = 10.vh
                }
                ref {
                    state.glyph?:return@ref
                    val canvas = it as? HTMLCanvasElement?:return@ref
                    val ctx = canvas.getContext("2d")?:return@ref
                    ctx.asDynamic().clearRect(0, 0, canvas.width, canvas.height)
                    val x = 40
                    val y = 150
                    val fontSize = 120
                    state.glyph?.draw(ctx, x, y, fontSize)
                    //state.glyph?.drawPoints(ctx, x, y, fontSize)
                    state.glyph?.drawMetrics(ctx, x, y, fontSize)
                }
            }
        }
    }

}

private fun RBuilder.char(
        groupKey: Int, unicode: Int,
        onGetGlyph: (Int, (OpentypeJS.Glyph?)->Unit)->Unit,
        onGetGlyphNow: (Int, (OpentypeJS.Glyph?)->Unit)->Unit
){
    child(Char::class){
        attrs {
            this.groupKey = groupKey
            this.unicode = unicode
            this.onGetGlyph = onGetGlyph
            this.onGetGlyphNow = onGetGlyphNow
        }
    }
}

interface CharGridProps : RProps {
    var onGetChars: ()->ArrayList<OpentypeJS.Glyph>
}

interface CharGridState : RState {
    var showingUnicodes: ArrayList<Int>
}

private class CharGrid : RComponent<CharGridProps, CharGridState>() {
    override fun CharGridState.init() {
        showingUnicodes = ArrayList()
    }

    override fun componentDidMount() {

    }

    private class OnGetGlyphRequest(val unicode: Int, val onGet: (OpentypeJS.Glyph?)->Unit)

    private var onGetGlyphRequests = ArrayList<OnGetGlyphRequest>()

    private var processOnGetGlyphRequestsTimer: Int? = null

    private val processOnGetGlyphRequests = fun(){
        //
        onGetGlyphRequests.sortBy{ it.unicode }
        //
        val requests = onGetGlyphRequests.filterIndexed(fun(index, onGetGlyphRequest): Boolean{
            val frontElement = onGetGlyphRequests.getOrNull(index-1)?:return true
            return frontElement.unicode != onGetGlyphRequest.unicode
        })
        //
        var i = 0
        var j = 0
        while (i < requests.size){
            val onGetGlyphRequest = requests.getOrNull(i)?:break
            val char = props.onGetChars().getOrNull(j)?:break
            val unicodeI = onGetGlyphRequest.unicode
            val unicodeJ = char.unicode?:0
            if(unicodeI == unicodeJ){
                onGetGlyphRequest.onGet(char)
                i++
                j++
            }else if(unicodeI < unicodeJ){
                onGetGlyphRequest.onGet(null)
                i++
            }else{
                j++
            }
        }
        while (i < requests.size){
            val onGetGlyphRequest = requests.getOrNull(i)?:break
            onGetGlyphRequest.onGet(null)
            i++
        }
        //
        onGetGlyphRequests.clear()
        processOnGetGlyphRequestsTimer = null
    }

    private val onGetGlyphByUnicode = fun(unicode: Int, onGet: (OpentypeJS.Glyph?)->Unit){
        onGetGlyphRequests.add(OnGetGlyphRequest(unicode, onGet))
        processOnGetGlyphRequestsTimer = processOnGetGlyphRequestsTimer?:window.setTimeout(
                fun(){processOnGetGlyphRequests()}, 5000
        )
    }

    private val onGetGlyphByUnicodeNow = fun(unicode: Int, onGet: (OpentypeJS.Glyph?)->Unit){
        onGetGlyphRequests.add(OnGetGlyphRequest(unicode, onGet))
        window.clearTimeout(processOnGetGlyphRequestsTimer?:0)
        window.setTimeout(fun(){processOnGetGlyphRequests()}, 10)
    }

    private var processedUnicode = 0

    private var charsIndex = 0

    override fun RBuilder.render() {
        GridLayout{
            attrs{
                options = jsObject {
                    this.isEqualSize = true
                    this.isConstantSize = true
                    this.transitionDuration = 0.2
                    this.useRecycle = true
                    this.useFirstRender = true
                }
                layoutOptions = jsObject {
                    this.margin = 10
                    this.align = "center"
                }
                onAppend = fun(obj: dynamic){
                    obj.startLoading()
                    /*
                    val groupKey = ((obj.groupKey.toString()).toIntOrNull()?:0)+1
                    val chars = props.chars
                    for (i in 0..9){
                        val char = chars.getOrNull(charsIndex)
                        state.showingGroupKeys.add(groupKey)
                        state.showingUnicodes.add(processedUnicode)
                        if(processedUnicode == char?.unicode){
                            state.showingChars.add(char)
                            charsIndex++
                        }else{
                            state.showingChars.add(null)
                        }
                        processedUnicode++
                    }
                    setState{}

                     */
                }
                onLayoutComplete = fun(obj: dynamic){
                    js("!obj.isLayout && obj.endLoading()")
                }
            }
            for (i  in 0..49999){
                char(i / 100, i, onGetGlyphByUnicode, onGetGlyphByUnicodeNow)
            }
            //+"${state.showingChars}"
        }
    }
}

fun RBuilder.charGrid(onGetChars: ()->ArrayList<OpentypeJS.Glyph>) = child(CharGrid::class){
    attrs.onGetChars = onGetChars
}

fun RBuilder.charGrid(onGetChars: ()->Array<OpentypeJS.Glyph>) = child(CharGrid::class){
    attrs.onGetChars = fun(): ArrayList<OpentypeJS.Glyph>{
        val arrayList = ArrayList<OpentypeJS.Glyph>()
        arrayList.addAll(onGetChars())
        return arrayList
    }
}