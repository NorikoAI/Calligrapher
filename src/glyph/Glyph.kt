package org.sourcekey.NorikoAI.Calligrapher

import kotlinx.html.classes
import org.sourcekey.NorikoAI.Calligrapher.OpentypeJS
import react.RBuilder
import react.dom.div
import react.dom.svg

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