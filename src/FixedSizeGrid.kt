package react.window

import kotlinext.js.Object
import kotlinx.html.DIV
import react.*
import react.dom.RDOMBuilder
import react.dom.div
import ticker.Ticker



// way 2


interface FixedSizeGridProps: RProps {
    var className : String
    var columnCount: Int
    var columnWidth: Int
    var direction: String
    var height: Int
    var initialScrollLeft: Int
    var initialScrollTop: Int
    var innerRef: ()->Unit
    var innerElementType: ReactElement
    var innerTagName: String
    var itemData: dynamic
    var itemKey: (columnIndex: Int, data: dynamic, rowIndex: Int)->String
    var onItemsRendered: (
            overscanColumnStartIndex: dynamic,
            overscanColumnStopIndex: dynamic,
            overscanRowStartIndex: dynamic,
            overscanRowStopIndex: dynamic,
            visibleColumnStartIndex: dynamic,
            visibleColumnStopIndex: dynamic,
            visibleRowStartIndex: dynamic,
            visibleRowStopIndex: dynamic
    )->Unit
    var onScroll: (
            horizontalScrollDirection: dynamic,
            scrollLeft: dynamic,
            scrollTop: dynamic,
            scrollUpdateWasRequested: dynamic,
            verticalScrollDirection: dynamic
    )->Unit
    var outerRef: ()->Unit
    var outerElementType: ReactElement
    var outerTagName: String
    var overscanColumnCount: Int
    var overscanColumnsCount: Int
    var overscanCount: Int
    var overscanRowCount: Int
    var overscanRowsCount: Int
    var rowCount: Int
    var rowHeight: Int
    var style: Object
    var useIsScrolling: Boolean
    var width: Int
}

// way 1
@JsModule("react-window")
@JsNonModule
@JsName("FixedSizeGrid")
external val FixedSizeGrid : RClass<FixedSizeGridProps>
/*
@JsModule("react-window")
@JsNonModule
@JsName("FixedSizeGrid")
external class FixedSizeGrid : Component<FixedSizeGridProps, RState> {
    override fun render(): ReactElement?
}
*/

fun RBuilder.fixedSizeGrid(block: RBuilder.() -> Unit) = FixedSizeGrid{
    attrs.columnCount = 10
    attrs.columnWidth = 10
    attrs.height = 1000
    attrs.rowCount = 10
    attrs.rowHeight = 10
    attrs.width = 1000
}
