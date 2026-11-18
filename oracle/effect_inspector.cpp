// Read-only access to the choice metadata of the pinned oracle. No rule or script is modified.
#include <iterator>
#include "duel.h"
#include "field.h"
#include "effect.h"
#include "card.h"
extern "C" int YGO_EffectEvent(void* raw, unsigned index, unsigned* code, unsigned* event) {
    if(!raw) return 0;
    const auto& choices = static_cast<duel*>(raw)->game_field->core.select_chains;
    if(index >= choices.size()) return 0;
    auto it = choices.begin();
    std::advance(it, index);
    if(!it->triggering_effect || !it->triggering_effect->handler) return 0;
    *code = it->triggering_effect->handler->data.code;
    *event = it->triggering_effect->code;
    return 1;
}

extern "C" int YGO_EffectDescription(void* raw, unsigned index, unsigned* code, uint64_t* description) {
    if(!raw) return 0;
    const auto& choices = static_cast<duel*>(raw)->game_field->core.select_chains;
    if(index >= choices.size()) return 0;
    auto it = choices.begin();
    std::advance(it, index);
    if(!it->triggering_effect || !it->triggering_effect->handler) return 0;
    *code = it->triggering_effect->handler->data.code;
    *description = it->triggering_effect->description;
    return 1;
}

// Identity of the cards in a location (cardid, creation order), read-only: out[i] for the i-th card of the location
// in sequence order (hand, deck, GY, banished, Extra Deck), or for zone i (Monster and Spell/Trap Zones;
// 0xFFFFFFFF for an empty zone). Returns the number of entries written.
extern "C" int YGO_CardIds(void* raw, unsigned player, unsigned location, unsigned* out, unsigned max) {
    if(!raw || player > 1) return 0;
    const auto& p = static_cast<duel*>(raw)->game_field->player[player];
    const card_vector* list = nullptr;
    switch(location) {
    case 0x01: list = &p.list_main; break;
    case 0x02: list = &p.list_hand; break;
    case 0x04: list = &p.list_mzone; break;
    case 0x08: list = &p.list_szone; break;
    case 0x10: list = &p.list_grave; break;
    case 0x20: list = &p.list_remove; break;
    case 0x40: list = &p.list_extra; break;
    default: return 0;
    }
    unsigned n = 0;
    for(const card* c : *list) {
        if(n >= max) break;
        out[n++] = c ? c->cardid : 0xFFFFFFFFu;
    }
    return static_cast<int>(n);
}

// Xyz materials (cardid, in order) of the monster in Monster Zone seq of player `player`.
extern "C" int YGO_OverlayIds(void* raw, unsigned player, unsigned seq, unsigned* out, unsigned max) {
    if(!raw || player > 1) return 0;
    const auto& p = static_cast<duel*>(raw)->game_field->player[player];
    if(seq >= p.list_mzone.size() || !p.list_mzone[seq]) return 0;
    unsigned n = 0;
    for(const card* c : p.list_mzone[seq]->xyz_materials) {
        if(n >= max) break;
        out[n++] = c->cardid;
    }
    return static_cast<int>(n);
}
