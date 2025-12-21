/**
 * @file circular_buffer.c
 * @brief Lock-free circular buffer implementation
 */

#include "circular_buffer.h"
#include <string.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 *═══════════════════════════════════════════════════════════════════════════════*/

void cbuf_init(CircularBuffer *buf)
{
    if (!buf)
        return;

    memset(buf->data, 0, sizeof(buf->data));
    CBUF_ATOMIC_STORE(&buf->head, 0);
    CBUF_ATOMIC_STORE(&buf->tail, 0);
    buf->total_pushed = 0;
    buf->total_dropped = 0;
    buf->total_consumed = 0;
}

void cbuf_reset(CircularBuffer *buf)
{
    if (!buf)
        return;

    CBUF_ATOMIC_STORE(&buf->head, 0);
    CBUF_ATOMIC_STORE(&buf->tail, 0);
    /* Keep statistics */
}

/*═══════════════════════════════════════════════════════════════════════════════
 * MAIN THREAD OPERATIONS (Producer)
 *═══════════════════════════════════════════════════════════════════════════════*/

bool cbuf_push(CircularBuffer *buf, double y, uint64_t tick_id)
{
    if (!buf)
        return false;

    uint32_t head = (uint32_t)CBUF_ATOMIC_LOAD(&buf->head);
    uint32_t tail = (uint32_t)CBUF_ATOMIC_LOAD(&buf->tail);

    uint32_t next_head = (head + 1) & CBUF_MASK;
    bool dropped = false;

    /* Check if full */
    if (next_head == tail)
    {
        /* Buffer full: drop oldest by advancing tail */
        uint32_t new_tail = (tail + 1) & CBUF_MASK;
        CBUF_ATOMIC_STORE(&buf->tail, new_tail);
        buf->total_dropped++;
        dropped = true;
    }

    /* Write entry */
    buf->data[head].y = y;
    buf->data[head].tick_id = tick_id;

    /* Advance head */
    CBUF_ATOMIC_STORE(&buf->head, next_head);
    buf->total_pushed++;

    return !dropped;
}

uint32_t cbuf_count(const CircularBuffer *buf)
{
    if (!buf)
        return 0;

    uint32_t head = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->head);
    uint32_t tail = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->tail);

    return (head - tail) & CBUF_MASK;
}

bool cbuf_is_full(const CircularBuffer *buf)
{
    return cbuf_count(buf) == (CBUF_CAPACITY - 1);
}

bool cbuf_is_empty(const CircularBuffer *buf)
{
    if (!buf)
        return true;

    uint32_t head = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->head);
    uint32_t tail = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->tail);

    return head == tail;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * BACKGROUND THREAD OPERATIONS (Consumer)
 *═══════════════════════════════════════════════════════════════════════════════*/

int cbuf_snapshot(const CircularBuffer *buf, BufferSnapshot *snapshot)
{
    if (!buf || !snapshot)
        return 0;

    uint32_t head = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->head);
    uint32_t tail = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->tail);

    int count = (int)((head - tail) & CBUF_MASK);
    snapshot->count = count;

    if (count == 0)
    {
        snapshot->first_tick = 0;
        snapshot->last_tick = 0;
        return 0;
    }

    /* Copy entries in order (oldest to newest) */
    for (int i = 0; i < count; i++)
    {
        uint32_t idx = (tail + (uint32_t)i) & CBUF_MASK;
        snapshot->observations[i] = buf->data[idx].y;
        snapshot->tick_ids[i] = buf->data[idx].tick_id;
    }

    snapshot->first_tick = snapshot->tick_ids[0];
    snapshot->last_tick = snapshot->tick_ids[count - 1];

    return count;
}

void cbuf_consume(CircularBuffer *buf, int count)
{
    if (!buf || count <= 0)
        return;

    uint32_t tail = (uint32_t)CBUF_ATOMIC_LOAD(&buf->tail);
    uint32_t new_tail = (tail + (uint32_t)count) & CBUF_MASK;

    CBUF_ATOMIC_STORE(&buf->tail, new_tail);
    buf->total_consumed += (uint64_t)count;
}

bool cbuf_peek_oldest(const CircularBuffer *buf, ObservationEntry *entry)
{
    if (!buf || !entry || cbuf_is_empty(buf))
        return false;

    uint32_t tail = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->tail);
    *entry = buf->data[tail];

    return true;
}

bool cbuf_peek_newest(const CircularBuffer *buf, ObservationEntry *entry)
{
    if (!buf || !entry || cbuf_is_empty(buf))
        return false;

    uint32_t head = (uint32_t)CBUF_ATOMIC_LOAD(&((CircularBuffer *)buf)->head);
    uint32_t newest = (head - 1) & CBUF_MASK;
    *entry = buf->data[newest];

    return true;
}

/*═══════════════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 *═══════════════════════════════════════════════════════════════════════════════*/

void cbuf_get_stats(const CircularBuffer *buf,
                    uint64_t *pushed,
                    uint64_t *dropped,
                    uint64_t *consumed)
{
    if (!buf)
        return;

    if (pushed)
        *pushed = buf->total_pushed;
    if (dropped)
        *dropped = buf->total_dropped;
    if (consumed)
        *consumed = buf->total_consumed;
}

uint64_t cbuf_get_staleness(const CircularBuffer *buf, uint64_t current_tick)
{
    if (!buf || cbuf_is_empty(buf))
        return 0;

    ObservationEntry oldest;
    if (!cbuf_peek_oldest(buf, &oldest))
        return 0;

    if (current_tick >= oldest.tick_id)
    {
        return current_tick - oldest.tick_id;
    }

    return 0;
}