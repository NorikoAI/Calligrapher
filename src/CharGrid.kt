package org.sourcekey.NorikoAI.Calligrapher

import kotlinext.js.jsObject
import kotlinx.css.*
import react.*
import react.dom.div
import react.dom.img
import react.dom.jsStyle
import react.dom.svg
import react.infinitegrid.GridLayout
import styled.css
import styled.styledDiv


external interface CharProps : RProps {
    var char: OpentypeJS.Glyph
}

fun RBuilder.char(unicode: Int, char: OpentypeJS.Glyph?): RClass<CharProps> {
    val item = rFunction<CharProps>("Char") { props ->
        styledDiv{
            css{
                if(char != null){ background = "#EEE" }else{ background = "#FEE" }
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

interface CharGridProps : RProps {
    var chars: ArrayList<OpentypeJS.Glyph>
}

interface CharGridState : RState {
    var showingChars: ArrayList<RClass<CharProps>>
}

class CharGrid : RComponent<CharGridProps, CharGridState>() {
    override fun CharGridState.init() {
        showingChars = ArrayList()
    }

    override fun componentDidMount() {

    }

    private var processedUnicode = 0

    private var charsIndex = 0

    override fun RBuilder.render() {
        GridLayout{
            attrs{
                options = jsObject {
                    isConstantSize = true
                    transitionDuration = 0.2
                    useRecycle = true
                    //isEqualSize = true
                }
                layoutOptions = jsObject {
                    margin = 10
                    align = "center"
                }
                onAppend = fun(obj: dynamic){
                    obj.startLoading()
                    val chars = props.chars
                    for (i in 0..9){
                        val char = chars.getOrNull(charsIndex)
                        if(processedUnicode == char?.unicode){
                            state.showingChars.add(char(processedUnicode, char))
                            charsIndex++
                        }else{
                            state.showingChars.add(char(processedUnicode, null))
                        }
                        processedUnicode++
                    }
                    setState{}
                }
                onLayoutComplete = fun(obj: dynamic){
                    js("!obj.isLayout && obj.endLoading()")
                }
            }
            for (char in state.showingChars){
                char.invoke{}
            }
            //+"${state.showingChars}"
        }
    }
}

fun RBuilder.charGrid(chars: ArrayList<OpentypeJS.Glyph>) = child(CharGrid::class){
    attrs.chars = chars
}

fun RBuilder.charGrid(chars: Array<OpentypeJS.Glyph>) = child(CharGrid::class){
    val arrayList = ArrayList<OpentypeJS.Glyph>()
    arrayList.addAll(chars)
    attrs.chars = arrayList
}