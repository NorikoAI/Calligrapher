package org.sourcekey.NorikoAI.Calligrapher

import OpentypeJS.Font
import OpentypeJS.load
import kotlinext.js.jsObject



data class Project(
        var name: String,
        private val fontUrl: String,
        private val referenceFontUrl: String = "font/SourceHanSans_v1.001/SourceHanSansTC-Regular.otf"
) {

    /**
     *
     * */
    var referenceFont: Font = run {
        load(referenceFontUrl).then { referenceFont = it }
        Font(jsObject {
            familyName = "null"
            styleName = "null"
            unitsPerEm = 1000
            ascender = 800
            descender = -200
        })
    }
    private set

    /**
     *
     * */
    var font: Font = run {
        load(fontUrl).then { font = it }
        Font(jsObject {
            familyName = "null"
            styleName = "null"
            unitsPerEm = 1000
            ascender = 800
            descender = -200
        })
    }
    private set

    val calligrapher = Calligrapher(this, fun():Font{return referenceFont}, fun():Font{return font})


}