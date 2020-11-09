package org.sourcekey.NorikoAI.Calligrapher

import OpentypeJS.*
import kotlinx.browser.window
import kotlinx.css.*
import kotlinx.html.weekInput
import org.w3c.dom.HTMLCanvasElement
import org.w3c.dom.RenderingContext
import react.*
import react.dom.*
import react.infinitegrid.*
import styled.css
import styled.styledCanvas
import styled.styledDiv


interface CharProps : RProps {
    var groupKey: Int
    var key: Int
    var unicode: Int
    var getFont: ()->OpentypeJS.Font
}

class CharState: RState{
    var glyph: OpentypeJS.Glyph? = null
}

private class Char: RComponent<CharProps, CharState>() {

    override fun CharState.init() {
        glyph = null
    }

    fun drawGlyph(){
        setState{
            this.glyph = try{
                val char = String.fromCodePoint(props.unicode)
                val glyph = props.getFont().charToGlyph(char)
                if(glyph.path.commands.isNotEmpty()){ glyph }else{ null }
            }catch(e: dynamic){ null }
        }
    }

    var timerID: Int? = null

    override fun componentDidMount() {
        drawGlyph()
        timerID = window.setInterval({ drawGlyph() }, 10000)
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
                    //state.glyph?.drawMetrics(ctx, x, y, fontSize)
                }
            }
        }
    }

}

private fun RBuilder.char(groupKey: Int, unicode: Int, getFont: ()->OpentypeJS.Font){
    child(Char::class){
        attrs {
            this.groupKey = groupKey
            this.unicode = unicode
            this.getFont = getFont
        }
    }
}

interface CharGridProps : RProps {
    var getFont: ()->OpentypeJS.Font
    var searchUnicode: Int
}

interface CharGridState : RState {
    var showingStartUnicode: Int
    var showingEndUnicode: Int
}

private class CharGrid(props: CharGridProps) : RComponent<CharGridProps, CharGridState>(props) {
    override fun CharGridState.init(props: CharGridProps) {
        showingStartUnicode = props.searchUnicode
        showingEndUnicode = props.searchUnicode
    }

    override fun componentDidMount() {

    }

    override fun componentWillUpdate(nextProps: CharGridProps, nextState: CharGridState) {
        state.showingStartUnicode = nextProps.searchUnicode
        state.showingEndUnicode = nextProps.searchUnicode
    }

    private val updateItemsEach = 100

    override fun RBuilder.render() {
        GridLayout{
            attrs{
                options = jsObject {
                    this.isEqualSize = true
                    this.isConstantSize = true
                    this.transitionDuration = 0.2
                    //this.useRecycle = true
                    //this.useFirstRender = true
                }
                layoutOptions = jsObject {
                    this.margin = 10
                    this.align = "center"
                }
                onPrepend = fun(e: OnPrependEvent){
                    //e.startLoading()
                    if(updateItemsEach <= state.showingStartUnicode){
                        setState{showingStartUnicode -= updateItemsEach}
                    }else{
                        if(0 < state.showingStartUnicode){
                            setState{showingStartUnicode = 0}
                        }else{ e.stop() }
                    }
                }
                onAppend = fun(e: OnAppendEvent){
                    //e.endLoading()
                    setState{showingEndUnicode += updateItemsEach}
                }
                onLayoutComplete = fun(e: OnLayoutCompleteEvent){
                    if(e.isLayout){e.endLoading()}
                }
            }
            for (i  in state.showingStartUnicode..state.showingEndUnicode){
                char(i/10, i, props.getFont)
            }
        }
    }
}

fun RBuilder.charGrid(getFont: ()->OpentypeJS.Font, searchUnicode: Int = 0) = child(CharGrid::class){
    attrs.getFont = getFont
    attrs.searchUnicode = searchUnicode
}

fun RBuilder.charGrid(getFont: ()->OpentypeJS.Font, searchUnicodeHex: String = "0") = child(CharGrid::class){
    attrs.getFont = getFont
    attrs.searchUnicode = searchUnicodeHex.toIntOrNull(16)?:0
}